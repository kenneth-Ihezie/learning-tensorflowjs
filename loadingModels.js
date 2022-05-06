const tf = require('@tensorflow/tfjs-node-gpu')
//This model predicts the price of a land using the land sqft
const MODEL_PATH = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';
let model;

async function loadModel(){
    model = await tf.loadLayersModel(MODEL_PATH)
    //to see the details of the model
    model.summary()

    //create a batch of 1...which is the land size
    const input = tf.tensor2d([[870]])
    //create a batch of 3....which is the land sizes
    const inputBatch = tf.tensor2d([[500], [1100], [970]])
    //actually the prediction for each batch
    const result = model.predict(input)
    const resultBatch = model.predict(inputBatch)

    result.print() //or use .array() to get results back as array
    resultBatch.print()  //or use .array() to get results back as array

    // always dispose the tensors you created
    input.dispose()
    inputBatch.dispose()
    result.dispose()
    resultBatch.dispose()
    model.dispose()
}

//works in the web brower
async function saveModelOffline(model){
    model.save('localstorage://demo/newModelName')
}

async function getModelFromOffline(){
    console.log(JSON.stringify(await tf.io.listModels()));
}
loadModel()
