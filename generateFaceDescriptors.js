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

const faceDescriptorsWritePath = "assets/faceDescriptors";

const labels = [];

async function init() {
  await loadFaceApiModels();

  fillLables();
  let faceApiDescriptors = await generatefaceDescriptors();
  console.log(faceApiDescriptors[0]);
  writeLabeledFaceDescriptorsToJson(faceApiDescriptors);
}

async function loadFaceApiModels() {
  //let res = await faceapi.loadFaceRecognitionModel(MODEL_URL); //model to Recognise Face
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
  return;
}

function fillLables() {
  // todo loop through /faces dir and make list that way
  for (let i = 1; i <= 101; i++) {
    // todo -> change I terug naar 101
    labels.push(`face${i}`);
  }
}

async function getlabeledFaceDescriptors() {
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

  return labeledFaceDescriptors;
}

function writeLabeledFaceDescriptorsToJson(labeledFaceDescriptors) {
  writeJson(labeledFaceDescriptors, faceDescriptorsWritePath, "faceDescriptors");
}

async function generatefaceDescriptors() {
  let labeledFaceDescriptors = await getlabeledFaceDescriptors();
  return labeledFaceDescriptors;
}

function writeJson(json, jsonWritePath, fileName) {
  let data = JSON.stringify(json, null, 4);
  fs.writeFileSync(jsonWritePath + "/" + fileName + ".json", data, "utf-8");
}

init();
