use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io;
use std::io::Read;

fn main() {
    let args: Vec<String> = env::args().collect();
    let events_file = if args.len() == 2 {
        &args[1]
    } else {
        eprintln!(
            "usage: {} EVENTS_FILE",
            args.first().map(String::as_ref).unwrap_or("rustboard")
        );
        std::process::exit(1);
    };
    println!("Processing file: {}", events_file);
    let accumulator = read_events(events_file).unwrap_or_else(|e| {
        eprintln!("error: {:?}", e);
        std::process::exit(1);
    });
    for (tag, points) in accumulator.time_series.iter() {
        println!();
        println!("=== {:?} ===", tag);
        for pt in points.iter() {
            println!("({}, {}) @ {}", pt.step, pt.value, pt.wall_time);
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TagId(String);

#[derive(Debug, PartialEq)]
struct ScalarPoint {
    /// Step index: unique non-negative key for this datum within its time series.
    step: i64,
    /// Wall time that the event was recorded, as seconds since epoch.
    wall_time: f64,
    /// Scalar value.
    value: f32,
}

struct ScalarsAccumulator {
    time_series: HashMap<TagId, Vec<ScalarPoint>>,
}

fn read_events(filename: &str) -> io::Result<ScalarsAccumulator> {
    let file = File::open(filename)?;
    let mut reader = io::BufReader::new(file);
    let mut result = ScalarsAccumulator {
        time_series: HashMap::new(),
    };
    loop {
        match read_event(&mut reader) {
            Ok(block) => parse_event_proto(&block, &mut result),
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(result)
}

/// Returns the read TF record, if possible.
fn read_event<R: Read>(f: &mut R) -> io::Result<Vec<u8>> {
    // From TensorFlow `record_writer.cc` comments:
    // Format of a single record:
    //  uint64    length
    //  uint32    masked crc of length
    //  byte      data[length]
    //  uint32    masked crc of data
    let len: u64 = f.read_u64::<LittleEndian>()?;
    f.read_u32::<LittleEndian>()?; // skip length checksum
    let mut buf: Vec<u8> = Vec::with_capacity(len as usize);
    if f.take(len).read_to_end(&mut buf)? < len as usize {
        return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
    }
    f.read_u32::<LittleEndian>()?; // skip data checksum
    Ok(buf)
}

struct ProtoKey {
    field_number: u64,
    _wire_type: WireType,
}

enum WireType {
    Varint,
    Fixed64,
    Fixed32,
    LengthDelimited,
}

impl ProtoKey {
    fn new(key: u64) -> ProtoKey {
        ProtoKey {
            field_number: (key & !0b111) >> 3,
            _wire_type: match key & 0b111 {
                0 => WireType::Varint,
                1 => WireType::Fixed64,
                2 => WireType::LengthDelimited,
                5 => WireType::Fixed32, // three, sir
                n => unimplemented!("wire type {}", n),
            },
        }
    }
    fn _skip_length(buf: &mut &[u8], n: usize) -> Option<()> {
        *buf = buf.get(n..)?;
        Some(())
    }
    fn skip(&self, buf: &mut &[u8]) -> Option<()> {
        match &self._wire_type {
            WireType::Varint => {
                read_varu64(buf)?;
                Some(())
            }
            WireType::LengthDelimited => {
                let len = read_varu64(buf)?;
                ProtoKey::_skip_length(buf, len as usize)
            }
            WireType::Fixed64 => ProtoKey::_skip_length(buf, 8),
            WireType::Fixed32 => ProtoKey::_skip_length(buf, 4),
        }
    }
    fn read<'a>(&self, buf: &mut &'a [u8]) -> Option<ProtoValue<'a>> {
        match &self._wire_type {
            WireType::Varint => Some(ProtoValue::Varint(read_varu64(buf)?)),
            WireType::Fixed64 => {
                if buf.len() < 8 {
                    None
                } else {
                    let result = LittleEndian::read_u64(buf);
                    ProtoKey::_skip_length(buf, 8);
                    Some(ProtoValue::Fixed64(result))
                }
            }
            WireType::Fixed32 => {
                if buf.len() < 4 {
                    None
                } else {
                    let result = LittleEndian::read_u32(buf);
                    ProtoKey::_skip_length(buf, 4);
                    Some(ProtoValue::Fixed32(result))
                }
            }
            WireType::LengthDelimited => {
                let len = read_varu64(buf)? as usize;
                if buf.len() < len {
                    return None;
                }
                let (value, rest) = buf.split_at(len);
                *buf = rest;
                Some(ProtoValue::LengthDelimited(value))
            }
        }
    }

    fn read_varint(&self, buf: &mut &[u8]) -> Option<u64> {
        match self.read(buf) {
            Some(ProtoValue::Varint(n)) => Some(n),
            _ => None,
        }
    }

    fn read_fixed64(&self, buf: &mut &[u8]) -> Option<u64> {
        match self.read(buf) {
            Some(ProtoValue::Fixed64(n)) => Some(n),
            _ => None,
        }
    }

    #[allow(unused)]
    fn read_fixed32(&self, buf: &mut &[u8]) -> Option<u32> {
        match self.read(buf) {
            Some(ProtoValue::Fixed32(n)) => Some(n),
            _ => None,
        }
    }

    fn read_length_delimited<'a>(&self, buf: &mut &'a [u8]) -> Option<&'a [u8]> {
        match self.read(buf) {
            Some(ProtoValue::LengthDelimited(result)) => Some(result),
            _ => None,
        }
    }
}

#[derive(Debug)]
enum ProtoValue<'a> {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    LengthDelimited(&'a [u8]),
}

fn parse_event_proto(event: &Vec<u8>, accumulator: &mut ScalarsAccumulator) {
    let mut buf: &[u8] = &event[..];
    let mut wall_time: f64 = 0.0;
    let mut step: i64 = 0;
    let mut tag_values: Vec<TagValue> = Vec::new();
    while let Some(()) = parse_event_field(&mut buf, &mut wall_time, &mut step, &mut tag_values) {}
    for tag_value in tag_values.into_iter() {
        accumulator
            .time_series
            .entry(tag_value.tag)
            .or_default()
            .push(ScalarPoint {
                step,
                wall_time,
                value: tag_value.value,
            })
    }

    // Relevant fields on `Event`:
    //   double wall_time = 1;
    //   int64 step = 2;
    //   string file_version = 3;
    //   Summary summary = 5;
    fn parse_event_field(
        buf: &mut &[u8],
        wall_time: &mut f64,
        step: &mut i64,
        tag_values: &mut Vec<TagValue>,
    ) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            1 => {
                if let Some(wall_time_bits) = key.read_fixed64(buf) {
                    *wall_time = f64::from_bits(wall_time_bits);
                }
            }
            2 => {
                if let Some(step_bits) = key.read_varint(buf) {
                    *step = step_bits as i64;
                }
            }
            5 => {
                if let Some(summary_msg) = key.read_length_delimited(buf) {
                    match parse_summary_proto(summary_msg) {
                        Some(tvs) => tag_values.extend(tvs.into_iter()),
                        None => (),
                    }
                }
            }
            _ => key.skip(buf)?,
        };
        Some(())
    }
}

struct TagValue {
    tag: TagId,
    value: f32,
}

fn parse_summary_proto(message: &[u8]) -> Option<Vec<TagValue>> {
    let mut buf: &[u8] = &message[..];
    let mut result = Vec::new();
    while let Some(()) = parse_summary_field(&mut buf, &mut result) {}
    return Some(result);

    // Relevant fields on `Summary`:
    //   repeated Value value = 1;
    fn parse_summary_field(buf: &mut &[u8], result: &mut Vec<TagValue>) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            1 => {
                if let Some(value_msg) = key.read_length_delimited(buf) {
                    result.extend(parse_value_proto(value_msg).into_iter())
                }
            }
            _ => key.skip(buf)?,
        };
        Some(())
    }
}

fn parse_value_proto(message: &[u8]) -> Option<TagValue> {
    let mut buf: &[u8] = &message[..];
    struct PartialTagValue {
        tag: Option<TagId>,
        value: Option<f32>,
        from_scalars_plugin: bool,
    }
    let mut result = PartialTagValue {
        tag: None,
        value: None,
        from_scalars_plugin: false,
    };
    while let Some(()) = parse_value_field(&mut buf, &mut result) {}
    return match result {
        PartialTagValue {
            tag: Some(tag),
            value: Some(value),
            from_scalars_plugin: true,
        } => Some(TagValue { tag, value }),
        _ => None,
    };

    // Relevant fields on `Value`:
    //   string tag = 1;
    //   SummaryMetadata metadata = 9;
    //   TensorProto tensor = 8;
    fn parse_value_field(buf: &mut &[u8], result: &mut PartialTagValue) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            1 => {
                if let Some(tag_bytes) = key.read_length_delimited(buf) {
                    let tag = String::from_utf8_lossy(tag_bytes).into_owned();
                    result.tag = Some(TagId(tag));
                }
            }
            8 => {
                if let Some(tensor_msg) = key.read_length_delimited(buf) {
                    result.value = parse_tensor_proto(tensor_msg).or(result.value);
                }
            }
            9 => {
                if let Some(metadata_msg) = key.read_length_delimited(buf) {
                    match parse_summary_metadata_proto(metadata_msg) {
                        Some(PluginName(ref s)) if s == "scalars" => {
                            result.from_scalars_plugin = true;
                        }
                        _ => (),
                    }
                }
            }
            _ => key.skip(buf)?,
        };
        Some(())
    }
}

fn parse_tensor_proto(message: &[u8]) -> Option<f32> {
    let mut buf: &[u8] = &message[..];
    let mut result: Option<f32> = None;
    while let Some(()) = parse_tensor_field(&mut buf, &mut result) {}
    return result;

    // On `Tensor`:
    //   DataType dtype = 1;
    //   TensorShapeProto tensor_shape = 2;
    //   bytes tensor_content = 4;
    //   repeated float float_val = 5 [packed = true];
    // On `DataType`:
    //   DT_FLOAT = 1;
    //   DT_DOUBLE = 2;
    fn parse_tensor_field(buf: &mut &[u8], result: &mut Option<f32>) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            4 | 5 => {
                // Field 4 is `tensor_content`, which we're assuming has DT_FLOAT data; field 5 is
                // packed `float_val`, so these two fields have the same representation and
                // interpretation.
                if let Some(tensor_content_bits) = key.read_length_delimited(buf) {
                    // Interpret as f32s.
                    for chunk in tensor_content_bits.chunks_exact(4) {
                        let val = f32::from_bits(LittleEndian::read_u32(chunk));
                        *result = Some(val);
                    }
                }
            }
            _ => key.skip(buf)?,
        };
        Some(())
    }
}

struct PluginName(String);

fn parse_summary_metadata_proto(message: &[u8]) -> Option<PluginName> {
    // Relevant fields on `SummaryMetadata`:
    //   PluginData plugin_data = 1;
    let mut buf: &[u8] = &message[..];
    let mut plugin_name: Option<PluginName> = None;
    while let Some(()) = parse_summary_metadata_field(&mut buf, &mut plugin_name) {}
    return plugin_name;

    fn parse_summary_metadata_field(
        buf: &mut &[u8],
        plugin_name: &mut Option<PluginName>,
    ) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            1 => {
                if let Some(plugin_data_msg) = key.read_length_delimited(buf) {
                    match parse_plugin_data_proto(plugin_data_msg) {
                        Some(v) => *plugin_name = Some(v),
                        None => (),
                    }
                }
            }
            _ => key.skip(buf)?,
        };
        Some(())
    }
}

fn parse_plugin_data_proto(message: &[u8]) -> Option<PluginName> {
    // Relevant fields on `PluginData`:
    //   string plugin_name = 1;
    let mut buf: &[u8] = &message[..];
    let mut result: Option<PluginName> = None;
    while let Some(()) = parse_plugin_data_field(&mut buf, &mut result) {}
    return result;

    fn parse_plugin_data_field(buf: &mut &[u8], result: &mut Option<PluginName>) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            1 => {
                if let Some(plugin_name_bits) = key.read_length_delimited(buf) {
                    let name = String::from_utf8_lossy(plugin_name_bits).into_owned();
                    *result = Some(PluginName(name));
                }
            }
            _ => key.skip(buf)?,
        };
        Some(())
    }
}

fn read_varu64(buf: &mut &[u8]) -> Option<u64> {
    let mut result: u64 = 0;
    for i in 0..9 {
        let byte = buf.get(0)?;
        *buf = &buf[1..];
        result |= ((byte & 0x7F) as u64) << (i * 7);
        if byte & 0x80 == 0 {
            return Some(result);
        }
    }
    None // took too many bytes, still wasn't done
}
