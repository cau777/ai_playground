# AI Playground

The project as a whole aims to build some small interactive playgrounds that use different AI techniques. At the moment,
it only supports handwritten digits recognition and chess, but I intend to add more projects later.

## More details
* [Digit recognition docs](https://github.com/cau777/ai_playground/tree/master/docs/digits)
* [Chess docs](https://github.com/cau777/ai_playground/tree/master/docs/chess)

### /codebase

This is a high-performant deep learning library made in Rust, that
uses [GPU shaders](https://github.com/cau777/ai_playground/tree/master/codebase/src/gpu/shaders) and CPU parallelization
to speed up computations. It's built almost from scratch and uses [ndarray](https://github.com/rust-ndarray/ndarray) for
n-dimensional arrays and [vulkano](https://github.com/vulkano-rs/vulkano) for GPU integration. It uses a more functional
approach (with centralized storage of parameters in a map), supporting:

* Convolution layer
  * Selective caching (used to improve the chess model performance by almost 85%)
* Max pool layer
* Dense layer
* Dropout layer
* Other utility layers
* ReLu, Tanh and Sigmoid activation functions
* Cross entropy and Mse loss functions
* Adam learning optimizer

A trained AI model can be represented in 3 files:
1) A binary file containing all parameters
2) A JSON file containing some info
3) A simple XML file defining the structure. For example:

```xml
<AIModel>
    <LossFunc><CrossEntropy/></LossFunc>
    <Layer>
        <Sequential>
            <Convolution in_channels="1" out_channels="32" kernel_size="5" stride="1" padding="2">
                <KernelsLr>
                    <Adam/>
                </KernelsLr>
            </Convolution>
            <Relu/>
            <MaxPool size="2" stride="2"/>
            {...}
        </Sequential>
    </Layer>
</AIModel>
```

Most of the time, changing the structure (adding or removing layers) does not lose any training progress.

### /client

A single page app built using React and Vite to allow users to interact with the playground in any device. It's hosted
in GitHub Pages and supports translations to English and Portuguese.

![Digit recognition page](https://github.com/cau777/ai_playground/blob/master/docs/screenshots/digits_page.png)

### /versions_server

Simple server built with [warp](https://github.com/seanmonstar/warp) to evaluate client requests using the trained
models, and manage AI versions (snapshots after a number of training epochs) and config files. It's hosted in Azure,
as a container mounted in a File Share.

### /trainer
Basic program to locally train a specific model and upload the results to *versions_server*.
