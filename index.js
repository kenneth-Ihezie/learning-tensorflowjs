const tf = require('@tensorflow/tfjs-node-gpu')

//creating array in tensor
let tensor = tf.tensor([1, 2, 3, 4])
tensor.print();
//one dimension tensor or rank one tensor
let oneD = tf.tensor1d([0,1,2])
//two dimesion tensor or rank two tensor
let twoD = tf.tensor2d([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])

let threeD = tf.tensor3d(
    [
        [
            [1,2,3],[4,5,6],[7,8,9] //first layer of 2d values
        ],
        [
            [0,0,0],[0,0,0],[0,0,0] //second layer of 2d values
        ],
        [
            [1,4,3],[4,5,6],[7,8,9] //third layer of 2d values
        ],
        [
            [1,4,3],[4,5,6],[7,8,9] //third layer of 2d values
        ],
        [
            [1,4,3],[4,5,6],[7,8,9] //third layer of 2d values
        ],
        [
            [1,4,3],[4,5,6],[7,8,9] //third layer of 2d values
        ],
    ],
)
console.log(threeD.print());


//Tensor Operation....
let data = tf.tensor2d([[1, 2, 3], [4, 5, 6]])
// Create a Tensor holding a single scalar value:
let scalar = tf.scalar(2)
// Multiply all values in tensor by the scalar value 2.
let newTensor = data.mul(scalar)
newTensor.print();

// You can even change the shape of the Tensor.
// This would convert the 2D tensor to a 1D version with
// 6 elements instead of a 2D 2 by 3 shape.
let reShaped = data.reshape([6])
reShaped.print();
/*
Note: We use methods such as ‘tensor.mul()’ for multiplication 
instead of regular JavaScript because we can leverage the faster 
execution of the graphics card or other backends that 
TensorFlow.js supports. These backends perform calculations 
in parallel and are much faster than vanilla JavaScript operations. 
*/

/*
Unlike regular JavaScript bindings, you need to dispose 
of tensors manually. If you skip this step, you could 
cause a memory leak that could slow down, or even crash 
your computer. The TensorFlow.js API provides special 
disposal methods such as tf.dispose and tf.tidy
*/
