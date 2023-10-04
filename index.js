// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
import "@tensorflow/tfjs-node";

import * as canvas from "canvas";

// import * as faceapi from "face-api.js";
import * as faceapi from "@vladmandic/face-api";

import fetch from "node-fetch";

import fs from "fs-extra";

const MODEL_URL = "assets/models"; //model directory

// Make face-api.js use that fetch implementation
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ fetch: fetch, Canvas, Image, ImageData });
const facesUrl = "http://127.0.0.1:5500/faces";
const testFacesUrl = "http://127.0.0.1:5500/faces_test";

const faceDescriptorsWritePath = "assets/faceDescriptors";

//let res = await faceapi.loadFaceRecognitionModel(MODEL_URL); //model to Recognise Face
await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);

const testFaceName = "Jorre.jpg";
const testFace = await canvas.loadImage(`${testFacesUrl}/${testFaceName}`); // todo - load from disk

let testFaceDescriptions = await faceapi
  .detectAllFaces(testFace)
  .withFaceLandmarks()
  .withFaceDescriptors();

let testFaceDescription = testFaceDescriptions[0];

testFaceDescriptions = faceapi.resizeResults(testFaceDescriptions, testFace);

const labels = [];

function fillLables() {
  // todo loop through /faces dir and make list that way
  for (let i = 1; i <= 101; i++) {
    // todo -> change I terug naar 101
    labels.push(`face${i}`);
  }
}

fillLables();

//const detections = await faceapi.detectAllFaces(img);

// console.log(detections);

/*
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

    let descriptorAsRegularArray = Array.from(faceDescription.descriptor);

    const faceDescriptors = [descriptorAsRegularArray];

    return {
      label: label,
      descriptors: faceDescriptors,
    };
    //return new faceapi.LabeledFaceDescriptors(label, faceDescriptors);
  })
);
*/

// writeJson(labeledFaceDescriptors, faceDescriptorsWritePath, "faceDescriptors");

// let labeledFaceDescriptors = fs.readJsonSync(`${faceDescriptorsWritePath}/faceDescriptors.json`);
// console.log(labeledFaceDescriptors);

let labeledFaceDescriptors_raw = fs.readFileSync(
  `${faceDescriptorsWritePath}/faceDescriptors.json`,
  "utf8"
);

let labeledFaceDescriptors = JSON.parse(labeledFaceDescriptors_raw);

function generateFaceApiDescriptors(labeledFaceDescriptors) {
  let faceApiDescriptors = [];

  for (let i = 0; i < labeledFaceDescriptors.length; i++) {
    let faceApiDescriptor = new faceapi.LabeledFaceDescriptors(
      labeledFaceDescriptors[i].label,
      labeledFaceDescriptors[i].descriptors
    );
    faceApiDescriptors.push(faceApiDescriptor);
  }

  return faceApiDescriptors;
}

function convertDescriptorsArrayToFloat32(originalArray) {
  let updatedArray = [];

  for (let i = 0; i < originalArray.length; i++) {
    let float32Arr = new Float32Array(originalArray[i].descriptors[0]);
    originalArray[i].descriptors[0] = float32Arr;
    updatedArray.push(originalArray[i]);
  }

  return updatedArray;
}

labeledFaceDescriptors = convertDescriptorsArrayToFloat32(labeledFaceDescriptors);

let faceApiDescriptors = generateFaceApiDescriptors(labeledFaceDescriptors);
findLookalike(faceApiDescriptors);

function findLookalike(faceApiDescriptors) {
  const threshold = 0.9;
  const faceMatcher = new faceapi.FaceMatcher(faceApiDescriptors, threshold);

  const results = testFaceDescriptions.map((fd) => faceMatcher.findBestMatch(fd.descriptor));

  console.log(results);
}

function writeJson(json, jsonWritePath, fileName) {
  // console.log(json);
  let data = JSON.stringify(json, null, 4);

  // console.log(data);

  fs.writeFileSync(jsonWritePath + "/" + fileName + ".json", data, "utf-8");
}
