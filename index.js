// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
import "@tensorflow/tfjs-node";

import * as canvas from "canvas";

// import * as faceapi from "face-api.js";
import * as faceapi from "@vladmandic/face-api";

import fetch from "node-fetch";

const MODEL_URL = "assets/models"; //model directory

// Make face-api.js use that fetch implementation
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ fetch: fetch, Canvas, Image, ImageData });

//let res = await faceapi.loadFaceRecognitionModel(MODEL_URL); //model to Recognise Face
await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL);
await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL);
await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL);

const myimg = await canvas.loadImage("http://127.0.0.1:5500/face_test.jpeg");

const detections = await faceapi.detectAllFaces(myimg).withFaceLandmarks().withFaceDescriptors();

console.log(detections);

let testFace = detections[0];

// let img = new Image();
// img.src = "http://127.0.0.1:5500/face_test.jpeg";

//const detections = await faceapi.detectAllFaces(img);

// console.log(detections);

//console.log(faceapi.nets);
