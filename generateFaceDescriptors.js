// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
import "@tensorflow/tfjs-node";

import * as canvas from "canvas";

// import * as faceapi from "face-api.js";
import * as faceapi from "@vladmandic/face-api";

import fetch from "node-fetch";

import fs from "fs-extra";
// const path = require("path");

import * as path from "path";

const MODEL_URL = "assets/models"; //model directory

// Make face-api.js use that fetch implementation
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ fetch: fetch, Canvas, Image, ImageData });

// const facesUrl = "http://127.0.0.1:5500/faces";

const faceDescriptorsWritePath = "assets/faceDescriptors";
const facesPath = "assets/faces_10k";
// const facesPath = "assets/faces_100";

let labels = [];

async function init() {
  console.time("generateTime");

  await loadFaceApiModels();

  labels = getFaceImgListFromDisk(facesPath);

  // labels = labels.slice(1, 3); // tmp test with a slice of array

  let faceApiDescriptors = await generatelabeledFaceDescriptors();

  console.log("** done generating **");

  writeLabeledFaceDescriptorsToJson(faceApiDescriptors);

  console.timeEnd("generateTime");
}

async function loadFaceApiModels() {
  //let res = await faceapi.loadFaceRecognitionModel(MODEL_URL); //model to Recognise Face
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);
  return;
}

function fillLables() {
  for (let i = 1; i <= 101; i++) {
    labels.push(`face${i}`);
  }
}

function getFaceImgListFromDisk(facesPath) {
  const jpgFiles = [];

  // Read the contents of the folder
  const files = fs.readdirSync(facesPath);

  // Loop through the files and filter for those with a .jpg extension
  files.forEach((file) => {
    const filePath = path.join(facesPath, file);

    // Check if the file is a .jpg image
    if (fs.statSync(filePath).isFile() && path.extname(file).toLowerCase() === ".jpg") {
      // Remove the .jpg extension and add the file name to the array
      const fileNameWithoutExtension = path.basename(file, ".jpg");
      jpgFiles.push(fileNameWithoutExtension);
    }
  });

  return jpgFiles;
}

async function generatelabeledFaceDescriptors() {
  const labeledFaceDescriptors = await Promise.all(
    labels.map(async (label, index) => {
      console.log("n = " + index);

      //const imgUrl = `${facesUrl}/${label}.jpg`;
      const imgUrl = `${facesPath}/${label}.jpg`;

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

function writeJson(json, jsonWritePath, fileName) {
  let data = JSON.stringify(json, null, 4);
  fs.writeFileSync(jsonWritePath + "/" + fileName + ".json", data, "utf-8");
}

init();
