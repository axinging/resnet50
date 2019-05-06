/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as resnet from 'tfjs-resnet'
resnet.hello();


import { resnetV2, resnetV1 } from 'tfjs-resnet'

import * as tf from '@tensorflow/tfjs'
//import { Cifar10 } from 'tfjs-cifar10-web'
import { Cifar10 } from 'tfjs-cifar10'
//import { ChartBatchLog, ChartEpochLog } from '../../utils/charts'


// const x = tf.randomNormal([3, 4, 4, 6])
// const A = resnetBlock(x, 2, [2, 4, 6], 1, 'a', 2)

//const model = resnetV1({ inputShape: [32, 32, 3], depth: 14 })
const model = resnetV2({ inputShape: [32, 32, 3], depth: 11 })

const optimizer = tf.train.adam()
model.compile({
  optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy']
})
model.summary()

//const chartBatch = new ChartBatchLog().init('mountNode')
//const chartEpoch = new ChartEpochLog().init('mountNode1')

async function train (data) { //Cifar10
  // The entire dataset doesn't fit into memory so we call train repeatedly
  // with batches using the fit() method.
  const { xs: x, ys: y } =  data.nextTrainBatch()
  console.log("111")
  const history = await model.fit(
    x.reshape([50000, 32, 32, 3]),{
      batchSize: 32,
      epochs: 200,
      callbacks: {
        onBatchEnd: (epoch, log) => {
          console.log('batch:', epoch, log)
          //if (epoch % 100 === 0) chartBatch.updata(log)
          return tf.nextFrame()
        },
        onEpochBegin: (epoch, log) => {
          console.time('Epoch training time')
          console.groupCollapsed('epoch times:', epoch)
          return tf.nextFrame()
        },
        onEpochEnd: (epoch, log) => {
          console.groupEnd()
          console.timeEnd('Epoch training time')
          console.log(epoch, log)

          const testBatch = data.nextTestBatch(2000)
          const score = model.evaluate(testBatch.xs.reshape([2000, 32, 32, 3]), testBatch.ys)
          console.timeEnd('Totol training time')
          score[0].print()
          score[1].print()

          //chartEpoch.updata(log)
          return tf.nextFrame()
        }
      }
    })
  console.log("222")
  await tf.nextFrame()
}

async function load () {
  resnet.hello();
  const data = new Cifar10()
  await data.load()
  console.log("333")
  await train(data)
  console.log("444")
  const {xs, ys} = data.nextTrainBatch(1500)
  console.log("555")
  console.log(xs, ys)
}

/*
function resnetLayer (
  { inputs, filters = 16, kernelSize = 3, strides = 1, activation = 'relu', batchNormalization = true, convFirst = true }:
  { inputs: tf.SymbolicTensor, filters?: number, kernelSize?: number, strides?: number, activation?: string, batchNormalization?: boolean, convFirst?: boolean }
): tf.SymbolicTensor {
  const conv = tf.layers.conv2d({ filters, kernelSize, strides, padding: 'same', kernelInitializer: tf.initializers.heNormal({}), kernelRegularizer: tf.regularizers.l2({l2: 1e-4}) })
  let x = inputs
  if (convFirst) {
    x = conv.apply(x) as tf.SymbolicTensor
    if (batchNormalization) x = tf.layers.batchNormalization({axis: 3}).apply(x) as tf.SymbolicTensor
    //FIXME:
    if (activation) x = tf.layers.activation({ activation: 'relu'  }).apply(x) as tf.SymbolicTensor
  } else {
    if (batchNormalization) x = tf.layers.batchNormalization({axis: 3}).apply(x) as tf.SymbolicTensor
    //FIXME:
    if (activation) x = tf.layers.activation({ activation: 'relu'  }).apply(x) as tf.SymbolicTensor
    x = conv.apply(x) as tf.SymbolicTensor
  }
  return x
}
*/

async function conv2dTest2() {
    const inputDepth = 1;
    const inShape = [2, 2, 2, inputDepth];
    const outputDepth = 1;
    const fSize = 1;
    const pad = 0;
    const stride = 1;

    const x = tf.tensor4d([1, 2, 3, 4, 5, 6, 7, 8], inShape);
    const w = tf.tensor4d([2], [fSize, fSize, inputDepth, outputDepth]);

    const result = tf.conv2d(x, w, stride, pad);
    //expect(result.shape).toEqual([2, 2, 2, 1]);
    const expected = [2, 4, 6, 8, 10, 12, 14, 16];
    //expectArraysClose(result, expected);
    result.print();
    console.log("***************************************");
    x = result.apply(x);
    x.print();
}

async function conv2dTest() {
    const inputShape = [32, 32, 3];
    const inputs = tf.input({ shape: inputShape });
    //const inputs = tf.ones([2, 2]);
    const filters = 16;
    const kernelSize = 3;
    const strides = 1;
    const activation = 'relu';
    const conv = tf.layers.conv2d({ filters, kernelSize, strides, padding: 'same', kernelInitializer: tf.initializers.heNormal({}), kernelRegularizer: tf.regularizers.l2({l2: 1e-4}) })
 
    console.log("***************************************");
    let x = inputs;
    //const res = conv.apply(x);
    //res.print();
}
conv2dTest();

ENV.set('WEBGL_PACK', false);
load();


