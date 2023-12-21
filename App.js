import * as tf from '@tensorflow/tfjs';
import {
  bundleResourceIO,
  cameraWithTensors,
} from '@tensorflow/tfjs-react-native';
import { Camera } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect, useRef, useState } from 'react';
import { Dimensions, StyleSheet, Text, View } from 'react-native';

const TensorCamera = cameraWithTensors(Camera);

// The size of camera preview.
//
// These are only for iOS devices.
const CAM_PREVIEW_WIDTH = Dimensions.get('window').width;
const CAM_PREVIEW_HEIGHT = CAM_PREVIEW_WIDTH / (9 / 16);

// The size of the output tensor (image) from TensorCamera.
//
// 9/16.
const OUTPUT_TENSOR_WIDTH = 224;
const OUTPUT_TENSOR_HEIGHT = 224;

export default function App() { 
  const [tfReady, setTfReady] = useState(false); 
  const [model, setModel] = useState();
  const [IsHotdog , setIsHotdog] = useState(null);


  const rafId = useRef(null);

  // Make sure tfjs and tfjs-react-native work, especially the tensor camera.
  useEffect(() => {
    async function prepare() {
      rafId.current = null;

      // Request camera permission.
      await Camera.requestCameraPermissionsAsync();

      // Wait for tfjs to initialize the backend.
      await tf.ready();

      // Load model.
      const modelJson = require('./model/model.json');
      const modelWeights = require('./model/weights.bin');
      
      const model = await tf.loadLayersModel(
        bundleResourceIO(modelJson, modelWeights)
      );
      setModel(model);
      //console.log(model)
      

      // Ready!!
      setTfReady(true);
    }

    prepare();
  }, []);

  // This will be called when the component in unmounted.
  useEffect(() => {
    return () => {
      if (rafId.current != null && rafId.current !== 0) {
        cancelAnimationFrame(rafId.current);
        rafId.current = 0;
      }
    };
  }, []);

  // Handler that will be called when TensorCamera is ready.
  const handleCameraStream = (images, updatePreview, gl) => {
    console.log('camera ready!');
    // Here, we want to get the tensor from each frame (image), and feed the
    // tensor to the model (which we will train separately).
    //
    // We will do this repeatly in a animation loop.
    const loop = () => {
      // This might not be necessary, but add it here just in case.
      if (rafId.current === 0) {
        return;
      }

      // Wrap this inside tf.tidy to release tensor memory automatically.
      tf.tidy(() => {
        //console.log('here is fine');
       
        const imageTensor = images.next().value;
        //console.log('here');

        const resize = tf.image.resizeBilinear(imageTensor, [224, 224]);
        
        const cropped = tf.expandDims(resize, 0); 
       
      

        // Feed the processed tensor to the model and get result tensor(s). 
        const result = model.predict(cropped); 
        //console.log('5');
        // Get the actual data (an array in this case) from the result tensor. 
        const logits = result.dataSync();
        console.log(logits);
        // Logits should be the probability of two classes (hot dog, not hot dog).
        if (logits) {
          setIsHotdog(logits[0] < logits[1]);
        } else {
          setIsHotdog(null);
        }
      });

      rafId.current = requestAnimationFrame(loop); 
    };

    loop();
  };

  if (!tfReady) {
    return (
      <View style={styles.loadingMsg}>
        <Text>Loading...</Text>
      </View>
    );
  } else {
    return (
      <View style={styles.container}>
        <TensorCamera
          style={styles.camera}
          autorender={true}
          type={Camera.Constants.Type.back}
          // Output tensor related props.
          // These decide the shape of output tensor from the camera.
          resizeWidth={OUTPUT_TENSOR_WIDTH}
          resizeHeight={OUTPUT_TENSOR_HEIGHT}
          resizeDepth={3}
          onReady={handleCameraStream} 
        />
        <View
          style={
            IsHotdog
              ? styles.resultContainernotMatch
              : styles.resultContainerMatch
          }
        >
          <Text style={styles.resultText}>
            {IsHotdog ? 'NotMatch' : 'Match'} 
          </Text>
        </View>
      </View>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    width: CAM_PREVIEW_WIDTH,
    height: CAM_PREVIEW_HEIGHT,  
    marginTop: Dimensions.get('window').height / 2 - CAM_PREVIEW_HEIGHT / 2, 
  },
  
  // Tensor camera requires z-index.
  camera: {
    width: '100%',
    height: '100%',
    zIndex: 1,
  },
  loadingMsg: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    alignItems: 'center',
    justifyContent: 'center',
  }, 
  resultContainerMatch: {
    position: 'absolute',
    top: 0,
    left: 0,
    zIndex: 100,
    padding: 20,
    borderRadius: 8,
    backgroundColor: '#00aa00',
  },
  resultContainernotMatch: {
    position: 'absolute', 
    top: 0,
    left: 0,
    zIndex: 100,
    padding: 20,
    borderRadius: 8, 
    backgroundColor: '#aa0000',
  },
  resultText: {
    fontSize: 30,
    color: 'white',
  },
}); 
