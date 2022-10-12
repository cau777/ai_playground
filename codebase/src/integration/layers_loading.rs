use crate::nn::{
    layers::nn_layers::Layer, loss::loss_func::LossFunc, lr_calculators::lr_calculator::LrCalc,
};
use std::{error::Error, fmt::Display};
use xmltree::Element;

#[derive(Debug)]
pub enum XmlError {
    ElementNotFound(&'static str),
    UnexpectedTag(String),
    AttributeNotFound(String, &'static str),
    AttributeParseError(String, &'static str, String),
    UnexpectedChildCount(String, u32, u32),
}

impl Display for XmlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ElementNotFound(e) => write!(f, "Element <{}> not found", e)?,
            Self::UnexpectedTag(e) => write!(f, "Unexpected tag <{}>", e)?,
            Self::AttributeNotFound(tag, name) => {
                write!(f, "Attribute '{}' not found in <{}>", name, tag)?
            }
            Self::AttributeParseError(tag, name, value) => write!(
                f,
                "Value '{}' isn't in the correct format for attribute '{}' in tag <{}>",
                value, name, tag
            )?,
            Self::UnexpectedChildCount(tag, expected, actual) => write!(
                f,
                "Tag <{}> is expected to have exactly {} children, but has {}",
                tag, expected, actual
            )?,
        }
        Ok(())
    }
}

impl Error for XmlError {}

type Result<T> = std::result::Result<T, XmlError>;

#[derive(Debug)]
pub struct ModelXmlData {
    pub loss_func: LossFunc,
    pub main_layer: Layer,
}

pub fn load_model_xml(bytes: &[u8]) -> Result<ModelXmlData> {
    let elements = xmltree::Element::parse_all(bytes).unwrap();

    let mut root = None;
    for e in iter_elements(&elements) {
        if e.name != "AIModel" {
            return Err(XmlError::UnexpectedTag(e.name.clone()));
        } else {
            root = Some(e)
        }
    }

    match root {
        Some(root) => {
            let children = &root.children;
            let mut loss_func = None;
            let mut main_layer = None;

            for e in iter_elements(children) {
                if e.name == "LossFunc" {
                    loss_func = Some(load_loss_func(e)?);
                } else if e.name == "Layer" {
                    main_layer = Some(load_main_layer(e)?);
                } else {
                    return Err(XmlError::UnexpectedTag(e.name.clone()));
                }
            }

            let loss_func = loss_func.ok_or_else(|| XmlError::ElementNotFound("LossFunc"))?;
            let main_layer = main_layer.ok_or_else(|| XmlError::ElementNotFound("Layer"))?;
            Ok(ModelXmlData {
                loss_func,
                main_layer,
            })
        }
        None => return Err(XmlError::ElementNotFound("AIModel")),
    }
}

fn load_loss_func(element: &Element) -> Result<LossFunc> {
    for e in iter_elements(&element.children) {
        match &e.name.as_str() {
            &"Mse" => return Ok(LossFunc::Mse),
            _ => return Err(XmlError::UnexpectedTag(e.name.clone())),
        }
    }
    Err(XmlError::ElementNotFound("Any loss function node"))
}

fn load_main_layer(element: &Element) -> Result<Layer> {
    let child = load_single_child(element)?;
    load_layer(child)
}

fn load_layer(element: &Element) -> Result<Layer> {
    use crate::nn::layers::*;
    match &element.name.as_str() {
        &"Sequential" => {
            let mut layers = Vec::new();
            for e in iter_elements(&element.children) {
                layers.push(load_layer(e)?)
            }
            Ok(Layer::Sequential(sequential_layer::SequentialLayerConfig {
                layers: layers,
            }))
        }
        &"Dense" => {
            // TODO: Init mode
            let weights_lr = iter_elements(&element.children)
                .find(|o| o.name == "WeightsLr")
                .ok_or_else(|| XmlError::ElementNotFound("WeightsLr"))?;
            let biases_lr = iter_elements(&element.children)
                .find(|o| o.name == "BiasesLr")
                .ok_or_else(|| XmlError::ElementNotFound("BiasesLr"))?;

            Ok(Layer::Dense(dense_layer::DenseLayerConfig {
                in_values: get_usize_attr(element, "in_values")?,
                out_values: get_usize_attr(element, "out_values")?,
                init_mode: dense_layer::DenseLayerInit::Random(),
                weights_lr_calc: load_lr(weights_lr)?,
                biases_lr_calc: load_lr(biases_lr)?,
            }))
        }
        &"Convolution" => {
            let kernels_lr = iter_elements(&element.children)
                .find(|o| o.name == "KernelsLr")
                .ok_or_else(|| XmlError::ElementNotFound("KernelsLr"))?;

            Ok(Layer::Convolution(convolution_layer::ConvolutionConfig {
                in_channels: get_usize_attr(element, "in_channels")?,
                out_channels: get_usize_attr(element, "out_channels")?,
                kernel_size: get_usize_attr(element, "kernel_size")?,
                stride: get_usize_attr(element, "stride")?,
                padding: get_usize_attr(element, "padding")?,
                init_mode: convolution_layer::ConvolutionInitMode::HeNormal(),
                lr_calc: load_lr(kernels_lr)?,
            }))
        }
        &"MaxPool" => {
            Ok(Layer::MaxPool(max_pool_layer::MaxPoolConfig {
                size: get_usize_attr(element, "size")?,
                stride: get_usize_attr(element, "stride")?,
            }))
        }
        _ => Err(XmlError::UnexpectedTag(element.name.clone())),
    }
}

fn load_lr(element: &Element) -> Result<LrCalc> {
    use crate::nn::lr_calculators::*;
    let element = load_single_child(element)?;
    match &element.name.as_str() {
        &"Adam" => {
            let mut config = adam_lr::AdamConfig::default();
            match get_f32_attr(element, "alpha") {
                Ok(v) => config.alpha = v,
                _ => {}
            };
            match get_f32_attr(element, "decay1") {
                Ok(v) => config.decay1 = v,
                _ => {}
            };
            match get_f32_attr(element, "decay2") {
                Ok(v) => config.decay2 = v,
                _ => {}
            };
            Ok(LrCalc::Adam(config))
        }
        &"Constant" => {
            let mut config = constant_lr::ConstantLrConfig::default();
            match get_f32_attr(element, "lr") {
                Ok(v) => config.lr = v,
                _ => {}
            };
            Ok(LrCalc::Constant(config))
        }
        _ => Err(XmlError::UnexpectedTag(element.name.clone())),
    }
}

fn load_single_child<'a>(element: &'a Element) -> Result<&'a Element> {
    let mut result = None;
    let mut count = 0;

    for e in iter_elements(&element.children) {
        result = Some(e);
        count += 1;
    }

    if count == 1 {
        Ok(result.unwrap())
    } else {
        Err(XmlError::UnexpectedChildCount(
            element.name.clone(),
            1,
            count,
        ))
    }
}

fn iter_elements<'a>(elements: &'a Vec<xmltree::XMLNode>) -> impl Iterator<Item = &'a Element> {
    elements.iter().filter_map(|o| o.as_element())
}

fn get_usize_attr(element: &Element, name: &'static str) -> Result<usize> {
    let value = element
        .attributes
        .get(name)
        .ok_or_else(|| XmlError::AttributeNotFound(element.name.clone(), name))?;

    value
        .parse()
        .map_err(|_| XmlError::AttributeParseError(element.name.clone(), name, value.clone()))
}

fn get_f32_attr(element: &Element, name: &'static str) -> Result<f32> {
    let value = element
        .attributes
        .get(name)
        .ok_or_else(|| XmlError::AttributeNotFound(element.name.clone(), name))?;

    value
        .parse()
        .map_err(|_| XmlError::AttributeParseError(element.name.clone(), name, value.clone()))
}

#[cfg(test)]
mod tests {
    use crate::nn::{layers::nn_layers::Layer};

    use super::load_model_xml;

    #[test]
    fn test1() {
        let str = r###"
<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<AIModel>
    <LossFunc>
        <Mse/>
    </LossFunc>
    <Layer>
        <Sequential>
            <Dense in_values="784" out_values="392">
                <WeightsLr>
                    <Adam/>
                </WeightsLr>
                <BiasesLr>
                    <Adam/>
                </BiasesLr>
            </Dense>
            
            <Dense in_values="392" out_values="10">
                <WeightsLr>
                    <Adam/>
                </WeightsLr>
                <BiasesLr>
                    <Adam/>
                </BiasesLr>
            </Dense>
        </Sequential>
    </Layer>
</AIModel>
"###;
        let result = load_model_xml(str.as_bytes()).unwrap();
        let mut correct = None;
        match result.main_layer {
            Layer::Sequential(l) => match &l.layers[0] {
                Layer::Dense(d) => correct = Some(d.in_values),
                _ => {}
            },
            _ => {}
        }

        assert_eq!(correct, Some(784))
    }
}
