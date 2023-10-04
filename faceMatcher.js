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

const testFaceName = "Jorre.jpg";

let testFace;

const labels = [];

async function init() {
  console.time("findMatchTime");

  await loadFaceApiModels();

  let testFaceDescriptions = await getTestFaceDescriptions();

  let faceApiDescriptors = getfaceDescriptorsFromJson();

  let lookalikeResult = findLookalike(testFaceDescriptions, faceApiDescriptors);
  console.log(lookalikeResult);

  console.timeEnd("findMatchTime");
}

async function loadFaceApiModels() {
  //let res = await faceapi.loadFaceRecognitionModel(MODEL_URL); //model to Recognise Face
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
  return;
}

async function getTestFaceDescriptions() {
  let testFace = await canvas.loadImage(`${testFacesUrl}/${testFaceName}`); // todo - load from disk

  let testFaceDescriptions = await faceapi
    .detectAllFaces(testFace)
    .withFaceLandmarks()
    .withFaceDescriptors();

  // let testFaceDescription = testFaceDescriptions[0];

  testFaceDescriptions = faceapi.resizeResults(testFaceDescriptions, testFace);

  return testFaceDescriptions;
}

function loadLabeledFaceDescriptorsFromDisk() {
  let labeledFaceDescriptors_raw = fs.readFileSync(
    `${faceDescriptorsWritePath}/faceDescriptors.json`,
    "utf8"
  );

  let labeledFaceDescriptors = JSON.parse(labeledFaceDescriptors_raw);

  labeledFaceDescriptors = convertDescriptorsArrayToFloat32(labeledFaceDescriptors);

  return labeledFaceDescriptors;
}

function convertFloatArrayDescriptorsToFaceApiDescriptors(labeledFaceDescriptors) {
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

function getfaceDescriptorsFromJson() {
  // plain parsed json faceDescriptors as array
  let labeledFaceDescriptors = loadLabeledFaceDescriptorsFromDisk();

  // convert to array of faceApi descriptors classes
  let faceApiDescriptors = convertFloatArrayDescriptorsToFaceApiDescriptors(labeledFaceDescriptors);

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

function findLookalike(testFaceDescriptions, faceApiDescriptors) {
  const threshold = 0.9;
  const faceMatcher = new faceapi.FaceMatcher(faceApiDescriptors, threshold);

  const results = testFaceDescriptions.map((fd) => faceMatcher.findBestMatch(fd.descriptor));

  return results;
}

function writeJson(json, jsonWritePath, fileName) {
  let data = JSON.stringify(json, null, 4);
  fs.writeFileSync(jsonWritePath + "/" + fileName + ".json", data, "utf-8");
}

init();
