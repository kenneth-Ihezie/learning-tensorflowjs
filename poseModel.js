const tf = require('@tensorflow/tfjs-node-gpu')
const MODEL_PATH = 'https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1'
let movenetModel;


async function loadAndRunModel(){
   movenetModel =  await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true })
   let exampleInputTensor = tf.zeros([1, 192, 192, 3], 'int32')
   let tensorOutput = movenetModel.predict(exampleInputTensor)
   let arrayOutput = await tensorOutput.array()

   console.log(arrayOutput);
}


loadAndRunModel()