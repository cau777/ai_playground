// This file is generated by rust-protobuf 3.2.0. Do not edit
// .proto file is parsed by protoc 3.19.4
// @generated

// https://github.com/rust-lang/rust-clippy/issues/702
#![allow(unknown_lints)]
#![allow(clippy::all)]

#![allow(unused_attributes)]
#![cfg_attr(rustfmt, rustfmt::skip)]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unused_results)]
#![allow(unused_mut)]

//! Generated file from `model_storage.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_3_2_0;

#[derive(PartialEq,Clone,Default,Debug)]
// @@protoc_insertion_point(message:proto_data.NdArrayData)
pub struct NdArrayData {
    // message fields
    // @@protoc_insertion_point(field:proto_data.NdArrayData.shape)
    pub shape: ::std::vec::Vec<u32>,
    // @@protoc_insertion_point(field:proto_data.NdArrayData.numbers)
    pub numbers: ::std::vec::Vec<f32>,
    // special fields
    // @@protoc_insertion_point(special_field:proto_data.NdArrayData.special_fields)
    pub special_fields: ::protobuf::SpecialFields,
}

impl<'a> ::std::default::Default for &'a NdArrayData {
    fn default() -> &'a NdArrayData {
        <NdArrayData as ::protobuf::Message>::default_instance()
    }
}

impl NdArrayData {
    pub fn new() -> NdArrayData {
        ::std::default::Default::default()
    }

    fn generated_message_descriptor_data() -> ::protobuf::reflect::GeneratedMessageDescriptorData {
        let mut fields = ::std::vec::Vec::with_capacity(2);
        let mut oneofs = ::std::vec::Vec::with_capacity(0);
        fields.push(::protobuf::reflect::rt::v2::make_vec_simpler_accessor::<_, _>(
            "shape",
            |m: &NdArrayData| { &m.shape },
            |m: &mut NdArrayData| { &mut m.shape },
        ));
        fields.push(::protobuf::reflect::rt::v2::make_vec_simpler_accessor::<_, _>(
            "numbers",
            |m: &NdArrayData| { &m.numbers },
            |m: &mut NdArrayData| { &mut m.numbers },
        ));
        ::protobuf::reflect::GeneratedMessageDescriptorData::new_2::<NdArrayData>(
            "NdArrayData",
            fields,
            oneofs,
        )
    }
}

impl ::protobuf::Message for NdArrayData {
    const NAME: &'static str = "NdArrayData";

    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::Result<()> {
        while let Some(tag) = is.read_raw_tag_or_eof()? {
            match tag {
                10 => {
                    is.read_repeated_packed_uint32_into(&mut self.shape)?;
                },
                8 => {
                    self.shape.push(is.read_uint32()?);
                },
                18 => {
                    is.read_repeated_packed_float_into(&mut self.numbers)?;
                },
                21 => {
                    self.numbers.push(is.read_float()?);
                },
                tag => {
                    ::protobuf::rt::read_unknown_or_skip_group(tag, is, self.special_fields.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u64 {
        let mut my_size = 0;
        for value in &self.shape {
            my_size += ::protobuf::rt::uint32_size(1, *value);
        };
        my_size += 5 * self.numbers.len() as u64;
        my_size += ::protobuf::rt::unknown_fields_size(self.special_fields.unknown_fields());
        self.special_fields.cached_size().set(my_size as u32);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::Result<()> {
        for v in &self.shape {
            os.write_uint32(1, *v)?;
        };
        for v in &self.numbers {
            os.write_float(2, *v)?;
        };
        os.write_unknown_fields(self.special_fields.unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn special_fields(&self) -> &::protobuf::SpecialFields {
        &self.special_fields
    }

    fn mut_special_fields(&mut self) -> &mut ::protobuf::SpecialFields {
        &mut self.special_fields
    }

    fn new() -> NdArrayData {
        NdArrayData::new()
    }

    fn clear(&mut self) {
        self.shape.clear();
        self.numbers.clear();
        self.special_fields.clear();
    }

    fn default_instance() -> &'static NdArrayData {
        static instance: NdArrayData = NdArrayData {
            shape: ::std::vec::Vec::new(),
            numbers: ::std::vec::Vec::new(),
            special_fields: ::protobuf::SpecialFields::new(),
        };
        &instance
    }
}

impl ::protobuf::MessageFull for NdArrayData {
    fn descriptor() -> ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::Lazy::new();
        descriptor.get(|| file_descriptor().message_by_package_relative_name("NdArrayData").unwrap()).clone()
    }
}

impl ::std::fmt::Display for NdArrayData {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for NdArrayData {
    type RuntimeType = ::protobuf::reflect::rt::RuntimeTypeMessage<Self>;
}

#[derive(PartialEq,Clone,Default,Debug)]
// @@protoc_insertion_point(message:proto_data.PairData)
pub struct PairData {
    // message fields
    // @@protoc_insertion_point(field:proto_data.PairData.key)
    pub key: ::std::string::String,
    // @@protoc_insertion_point(field:proto_data.PairData.arrays)
    pub arrays: ::std::vec::Vec<NdArrayData>,
    // special fields
    // @@protoc_insertion_point(special_field:proto_data.PairData.special_fields)
    pub special_fields: ::protobuf::SpecialFields,
}

impl<'a> ::std::default::Default for &'a PairData {
    fn default() -> &'a PairData {
        <PairData as ::protobuf::Message>::default_instance()
    }
}

impl PairData {
    pub fn new() -> PairData {
        ::std::default::Default::default()
    }

    fn generated_message_descriptor_data() -> ::protobuf::reflect::GeneratedMessageDescriptorData {
        let mut fields = ::std::vec::Vec::with_capacity(2);
        let mut oneofs = ::std::vec::Vec::with_capacity(0);
        fields.push(::protobuf::reflect::rt::v2::make_simpler_field_accessor::<_, _>(
            "key",
            |m: &PairData| { &m.key },
            |m: &mut PairData| { &mut m.key },
        ));
        fields.push(::protobuf::reflect::rt::v2::make_vec_simpler_accessor::<_, _>(
            "arrays",
            |m: &PairData| { &m.arrays },
            |m: &mut PairData| { &mut m.arrays },
        ));
        ::protobuf::reflect::GeneratedMessageDescriptorData::new_2::<PairData>(
            "PairData",
            fields,
            oneofs,
        )
    }
}

impl ::protobuf::Message for PairData {
    const NAME: &'static str = "PairData";

    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::Result<()> {
        while let Some(tag) = is.read_raw_tag_or_eof()? {
            match tag {
                10 => {
                    self.key = is.read_string()?;
                },
                18 => {
                    self.arrays.push(is.read_message()?);
                },
                tag => {
                    ::protobuf::rt::read_unknown_or_skip_group(tag, is, self.special_fields.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u64 {
        let mut my_size = 0;
        if !self.key.is_empty() {
            my_size += ::protobuf::rt::string_size(1, &self.key);
        }
        for value in &self.arrays {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint64_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.special_fields.unknown_fields());
        self.special_fields.cached_size().set(my_size as u32);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::Result<()> {
        if !self.key.is_empty() {
            os.write_string(1, &self.key)?;
        }
        for v in &self.arrays {
            ::protobuf::rt::write_message_field_with_cached_size(2, v, os)?;
        };
        os.write_unknown_fields(self.special_fields.unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn special_fields(&self) -> &::protobuf::SpecialFields {
        &self.special_fields
    }

    fn mut_special_fields(&mut self) -> &mut ::protobuf::SpecialFields {
        &mut self.special_fields
    }

    fn new() -> PairData {
        PairData::new()
    }

    fn clear(&mut self) {
        self.key.clear();
        self.arrays.clear();
        self.special_fields.clear();
    }

    fn default_instance() -> &'static PairData {
        static instance: PairData = PairData {
            key: ::std::string::String::new(),
            arrays: ::std::vec::Vec::new(),
            special_fields: ::protobuf::SpecialFields::new(),
        };
        &instance
    }
}

impl ::protobuf::MessageFull for PairData {
    fn descriptor() -> ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::Lazy::new();
        descriptor.get(|| file_descriptor().message_by_package_relative_name("PairData").unwrap()).clone()
    }
}

impl ::std::fmt::Display for PairData {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for PairData {
    type RuntimeType = ::protobuf::reflect::rt::RuntimeTypeMessage<Self>;
}

#[derive(PartialEq,Clone,Default,Debug)]
// @@protoc_insertion_point(message:proto_data.ModelStorageData)
pub struct ModelStorageData {
    // message fields
    // @@protoc_insertion_point(field:proto_data.ModelStorageData.pairs)
    pub pairs: ::std::vec::Vec<PairData>,
    // special fields
    // @@protoc_insertion_point(special_field:proto_data.ModelStorageData.special_fields)
    pub special_fields: ::protobuf::SpecialFields,
}

impl<'a> ::std::default::Default for &'a ModelStorageData {
    fn default() -> &'a ModelStorageData {
        <ModelStorageData as ::protobuf::Message>::default_instance()
    }
}

impl ModelStorageData {
    pub fn new() -> ModelStorageData {
        ::std::default::Default::default()
    }

    fn generated_message_descriptor_data() -> ::protobuf::reflect::GeneratedMessageDescriptorData {
        let mut fields = ::std::vec::Vec::with_capacity(1);
        let mut oneofs = ::std::vec::Vec::with_capacity(0);
        fields.push(::protobuf::reflect::rt::v2::make_vec_simpler_accessor::<_, _>(
            "pairs",
            |m: &ModelStorageData| { &m.pairs },
            |m: &mut ModelStorageData| { &mut m.pairs },
        ));
        ::protobuf::reflect::GeneratedMessageDescriptorData::new_2::<ModelStorageData>(
            "ModelStorageData",
            fields,
            oneofs,
        )
    }
}

impl ::protobuf::Message for ModelStorageData {
    const NAME: &'static str = "ModelStorageData";

    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::Result<()> {
        while let Some(tag) = is.read_raw_tag_or_eof()? {
            match tag {
                10 => {
                    self.pairs.push(is.read_message()?);
                },
                tag => {
                    ::protobuf::rt::read_unknown_or_skip_group(tag, is, self.special_fields.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u64 {
        let mut my_size = 0;
        for value in &self.pairs {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint64_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.special_fields.unknown_fields());
        self.special_fields.cached_size().set(my_size as u32);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::Result<()> {
        for v in &self.pairs {
            ::protobuf::rt::write_message_field_with_cached_size(1, v, os)?;
        };
        os.write_unknown_fields(self.special_fields.unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn special_fields(&self) -> &::protobuf::SpecialFields {
        &self.special_fields
    }

    fn mut_special_fields(&mut self) -> &mut ::protobuf::SpecialFields {
        &mut self.special_fields
    }

    fn new() -> ModelStorageData {
        ModelStorageData::new()
    }

    fn clear(&mut self) {
        self.pairs.clear();
        self.special_fields.clear();
    }

    fn default_instance() -> &'static ModelStorageData {
        static instance: ModelStorageData = ModelStorageData {
            pairs: ::std::vec::Vec::new(),
            special_fields: ::protobuf::SpecialFields::new(),
        };
        &instance
    }
}

impl ::protobuf::MessageFull for ModelStorageData {
    fn descriptor() -> ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::Lazy::new();
        descriptor.get(|| file_descriptor().message_by_package_relative_name("ModelStorageData").unwrap()).clone()
    }
}

impl ::std::fmt::Display for ModelStorageData {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for ModelStorageData {
    type RuntimeType = ::protobuf::reflect::rt::RuntimeTypeMessage<Self>;
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n\x13model_storage.proto\x12\nproto_data\"=\n\x0bNdArrayData\x12\x14\n\
    \x05shape\x18\x01\x20\x03(\rR\x05shape\x12\x18\n\x07numbers\x18\x02\x20\
    \x03(\x02R\x07numbers\"M\n\x08PairData\x12\x10\n\x03key\x18\x01\x20\x01(\
    \tR\x03key\x12/\n\x06arrays\x18\x02\x20\x03(\x0b2\x17.proto_data.NdArray\
    DataR\x06arrays\">\n\x10ModelStorageData\x12*\n\x05pairs\x18\x01\x20\x03\
    (\x0b2\x14.proto_data.PairDataR\x05pairsb\x06proto3\
";

/// `FileDescriptorProto` object which was a source for this generated file
fn file_descriptor_proto() -> &'static ::protobuf::descriptor::FileDescriptorProto {
    static file_descriptor_proto_lazy: ::protobuf::rt::Lazy<::protobuf::descriptor::FileDescriptorProto> = ::protobuf::rt::Lazy::new();
    file_descriptor_proto_lazy.get(|| {
        ::protobuf::Message::parse_from_bytes(file_descriptor_proto_data).unwrap()
    })
}

/// `FileDescriptor` object which allows dynamic access to files
pub fn file_descriptor() -> &'static ::protobuf::reflect::FileDescriptor {
    static generated_file_descriptor_lazy: ::protobuf::rt::Lazy<::protobuf::reflect::GeneratedFileDescriptor> = ::protobuf::rt::Lazy::new();
    static file_descriptor: ::protobuf::rt::Lazy<::protobuf::reflect::FileDescriptor> = ::protobuf::rt::Lazy::new();
    file_descriptor.get(|| {
        let generated_file_descriptor = generated_file_descriptor_lazy.get(|| {
            let mut deps = ::std::vec::Vec::with_capacity(0);
            let mut messages = ::std::vec::Vec::with_capacity(3);
            messages.push(NdArrayData::generated_message_descriptor_data());
            messages.push(PairData::generated_message_descriptor_data());
            messages.push(ModelStorageData::generated_message_descriptor_data());
            let mut enums = ::std::vec::Vec::with_capacity(0);
            ::protobuf::reflect::GeneratedFileDescriptor::new_generated(
                file_descriptor_proto(),
                deps,
                messages,
                enums,
            )
        });
        ::protobuf::reflect::FileDescriptor::new_generated_2(generated_file_descriptor)
    })
}
