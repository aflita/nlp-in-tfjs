let input = document.querySelector('input');
let button = document.querySelector('button');
button.addEventListener('click', onClick);
//let label = document.querySelector('p');


let isModelLoaded = false;

let model;
let word2index;

const maxlen = 18;
const vocab_size = 2000;
const padding = 'post';
const truncating = 'post';

var myVar;

function myFunction() {
    myVar = setTimeout(showPage, 3000);
}

function showPage() {
    document.getElementById("loaderlabel").style.display = "none";
    document.getElementById("loader").style.display = "none";       
    document.getElementById("mainAPP").style.display = "block";
}

function getInput(){
    const reviewText = document.getElementById('input')
    return reviewText.value;
}

function onClick(){
    
    if(!isModelLoaded) {
        alert('Model not loaded yet');
        return;
    }

    if (getInput() === '') {
        alert("Review Can't be Null");
        document.getElementById('input').focus();
        return;
    }
    
    const inputText = getInput().trim().toLowerCase().split(" ");
    let score = predict(inputText);
  
    if (score > 0.5) {
        alert ('Positive Review \n'+score);
    } else {
        alert ('Negative Review \n'+score);
    }
}

function predict(inputText){

    //return console.log(model.summary());
    //return console.log(word2index['food']);

    const sequence = inputText.map(word => {
        let indexed = word2index[word];

        if (indexed === undefined){
            return 1; //change to oov value
        }
        return indexed;
    });
    
    const paddedSequence = padSequence([sequence], maxlen);

    const score = tf.tidy(() => {
        const input = tf.tensor2d(paddedSequence, [1, maxlen]);
        const result = model.predict(input);
        return result.dataSync()[0];
    });

    return score;  

}

function padSequence(sequences, maxLen, padding='post', truncating = "post", pad_value = 0){
    return sequences.map(seq => {
        if (seq.length > maxLen) { //truncat
            if (truncating === 'pre'){
                seq.splice(0, seq.length - maxLen);
            } else {
                seq.splice(maxLen, seq.length - maxLen);
            }
        }
                
        if (seq.length < maxLen) {
            const pad = [];
            for (let i = 0; i < maxLen - seq.length; i++){
                pad.push(pad_value);
            }
            if (padding === 'pre') {
                seq = pad.concat(seq);
            } else {
                seq = seq.concat(pad);
            }
        }               
        return seq;
        });
}

async function init(){

    model = await tf.loadLayersModel('http://127.0.0.1:8887/model.json');
    isModelLoaded = true;

    const word2indexjson = await fetch('http://127.0.0.1:8887/word2index.json');
    word2index = await word2indexjson.json();
    console.log(model.summary());
    console.log('Model & Metadata Loaded Succesfully');
}

function detectWebGLContext () {
    // Create canvas element. The canvas is not added to the
    // document itself, so it is never displayed in the
    // browser window.
    var canvas = document.createElement("canvas");
    // Get WebGLRenderingContext from canvas element.
    var gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    // Report the result.
    if (gl && gl instanceof WebGLRenderingContext) {
        console.log("Congratulations! Your browser supports WebGL.");
        init();
    } else {
        alert("Failed to get WebGL context. Your browser or device may not support WebGL.");
    }
}

detectWebGLContext();
