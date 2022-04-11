import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js';

const CLOTHES={
  '0':'T-SHIRT',
  '1': 'TROUSERS',
  '2': 'PULLOVER',
  '3': 'DRESS',
  '4': 'COAT',
  '5': 'SANDAL',
  '6': 'SHIRT',
  '7': 'SNEAKERS',
  '8': 'BAG',
  '9': 'ANKLE BOOT'
}

// Grab a reference to the MNIST input values (pixel data).

const INPUTS = TRAINING_DATA.inputs;


// Grab reference to the MNIST output values.

const OUTPUTS = TRAINING_DATA.outputs;


// Shuffle the two arrays in the same way so inputs still match outputs indexes.

tf.util.shuffleCombo(INPUTS, OUTPUTS);


// Input feature Array is 2 dimensional.

const INPUTS_TENSOR = tf.tensor2d(INPUTS);


// Output feature Array is 1 dimensional.

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

// Now actually create and define model architecture.

const model = tf.sequential();

//Tenemos tres capas, la capa de entrada con 28*28 valores y 32 neuronas
//Las dos primeras capas toman la función de activación relu
//hidden layer con 16 neuronas y la final con 10 neuronas una por posible salida
//y función de activación típica para clasificación softmax:
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));

model.add(tf.layers.dense({units: 16, activation: 'relu'}));

model.add(tf.layers.dense({units: 10, activation: 'softmax'}));


model.summary();


train();

async function train() { 

    // Compile the model with the defined optimizer and specify our loss function to use.
  
    model.compile({
  
      optimizer: 'adam',
  
      loss: 'categoricalCrossentropy',
  
      metrics: ['accuracy'] //measure of how many images are predicted correctly from the training data.
  
    });

    function logProgress(epoch, logs) {

      console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));

    }
  
  
    let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
  
      shuffle: true,        // Ensure data is shuffled again before using each epoch.
  
      validationSplit: 0.1,
  
      batchSize: 250,       // Update weights after every 512 examples.      
  
      epochs: 50,           // Go over the data 50 times!

      callbacks: {onEpochEnd: logProgress},
  
  
    });
  
    
  
    OUTPUTS_TENSOR.dispose();
  
    INPUTS_TENSOR.dispose();

    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
  
    evaluate(); // Once trained we can evaluate the model.
  
  }

  const PREDICTION_ELEMENT = document.getElementById('prediction');


function evaluate() {

  const OFFSET = Math.floor((Math.random() * INPUTS.length)); // Select random from all example inputs. 

 

  let answer = tf.tidy(function() {

    let newInput = tf.tensor1d(INPUTS[OFFSET]).expandDims();

    

    let output = model.predict(newInput);

    output.print();

    return output.squeeze().argMax();    

  });

  answer.array().then(function(index) {

    PREDICTION_ELEMENT.innerText = CLOTHES[`${index}`];
    console.log(index, OUTPUTS[OFFSET])
    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');

    answer.dispose();

    drawImage(INPUTS[OFFSET]);

  });

}

const CANVAS = document.getElementById('canvas');

const CTX = CANVAS.getContext('2d');


function drawImage(digit) {

  var imageData = CTX.getImageData(0, 0, 28, 28); //Origen  0,0 y tamaño imagen

  

  for (let i = 0; i < digit.length; i++) {

    imageData.data[i * 4] = digit[i] * 255;      // Red Channel.

    imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.

    imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.

    imageData.data[i * 4 + 3] = 255;             // Alpha Channel.

  }


  // Render the updated array of data to the canvas itself.

  CTX.putImageData(imageData, 0, 0); 


  // Perform a new classification after a certain interval.

  setTimeout(evaluate, 20000);

}