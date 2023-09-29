// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
import "@tensorflow/tfjs-node";

import * as canvas from "canvas";

// import * as faceapi from "face-api.js";
import * as faceapi from "@vladmandic/face-api";

import fetch from "node-fetch";

const fs = require("fs-extra");

const MODEL_URL = "assets/models"; //model directory

// Make face-api.js use that fetch implementation
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ fetch: fetch, Canvas, Image, ImageData });
const facesUrl = "http://127.0.0.1:5500/faces";
const testFacesUrl = "http://127.0.0.1:5500/faces_test";

const faceDescriptorsWritePath = "assets/assets/faceDescriptors";

//let res = await faceapi.loadFaceRecognitionModel(MODEL_URL); //model to Recognise Face
await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);

const testFaceName = "Jorre.jpg";
const testFace = await canvas.loadImage(`${testFacesUrl}/${testFaceName}`); // todo - load from disk

let faceDescriptions = await faceapi
  .detectAllFaces(testFace)
  .withFaceLandmarks()
  .withFaceDescriptors();

let testFaceDescription = faceDescriptions[0];

faceDescriptions = faceapi.resizeResults(faceDescriptions, testFace);

// console.log(testFace);
// console.log(testFace);
//console.log(faceapi.nets);

// const labels = [
//   "face1",
//   "face2",
//   "face3",
//   "face4",
//   "face5",
//   "face6",
//   "face7",
//   "face8",
//   "face9",
//   "face10",
// ];

const labels = [];

function fillLables() {
  // todo loop through /faces dir and make list that way
  for (let i = 1; i <= 102; i++) {
    labels.push(`face${i}`);
  }
}

fillLables();

//const detections = await faceapi.detectAllFaces(img);

// console.log(detections);

const labeledFaceDescriptors = await Promise.all(
  labels.map(async (label) => {
    // const imgUrl = `images/${label}.jpeg`;
    const imgUrl = `${facesUrl}/${label}.jpg`;
    // const img = await faceapi.fetchImage(imgUrl);
    const img = await canvas.loadImage(imgUrl);

    const faceDescription = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();
    if (!faceDescription) {
      throw new Error(`no faces detected for ${label}`);
    }

    const faceDescriptors = [faceDescription.descriptor];
    return new faceapi.LabeledFaceDescriptors(label, faceDescriptors);
  })
);

// console.log(labeledFaceDescriptors);

//writeJson(labeledFaceDescriptors, "./logs", "rowsAdded", false);

// match test face
const threshold = 0.9;
const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, threshold);

const results = faceDescriptions.map((fd) => faceMatcher.findBestMatch(fd.descriptor));

console.log(results);

// log results
//results.forEach((bestMatch, i) => {
//console.log(bestMatch);
// const box = faceDescriptions[i].detection.box
// const text = bestMatch.toString();
//console.log(text);
// const drawBox = new faceapi.draw.DrawBox(box, { label: text })
// drawBox.draw(canvas)
//});

// function writeJson(json, jsonWritePath, fileName) {
//   let data = JSON.stringify(json, null, 4);
//   fs.writeFileSync(jsonWritePath + "/" + fileName + ".json", data);
// }
