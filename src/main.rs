use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
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
    read_events(events_file).unwrap_or_else(|e| {
        eprintln!("error: {:?}", e);
        std::process::exit(1);
    });
}

fn read_events(filename: &str) -> io::Result<()> {
    let mut file = File::open(filename)?;
    loop {
        match read_event(&mut file) {
            Ok(block) => parse_event_proto(&block),
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Returns the read TF record, if possible.
fn read_event(f: &mut File) -> io::Result<Vec<u8>> {
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
}

#[derive(Debug)]
enum ProtoValue<'a> {
    Varint(u64),
    Fixed64(u64),
    Fixed32(u32),
    LengthDelimited(&'a [u8]),
}

fn parse_event_proto(event: &Vec<u8>) {
    // Relevant fields on proto `Event`:
    //   double wall_time = 1;
    //   int64 step = 2;
    //   Summary summary = 5;
    // On `Summary`:
    //   repeated Value value = 1;
    // On `Value`:
    //   string tag = 1;
    //   SummaryMetadata metadata = 9;
    //   TensorProto tensor = 8;
    // On `SummaryMetadata`:
    //   PluginData plugin_data = 1;
    // On `PluginData`:
    //   string plugin_name = 1;
    // On `Tensor`:
    //   repeated double double_val = 6 [packed = true];
    let mut buf: &[u8] = &event[..];
    print!("event {{ ");
    while let Some(()) = parse_event_field(&mut buf) {}
    println!("}}");

    fn parse_event_field(buf: &mut &[u8]) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            1 => print!(
                "wall_time: {} ",
                match key.read(buf)? {
                    ProtoValue::Fixed64(n) => format!("{:?}", f64::from_bits(n)),
                    other => format!("unexpected[{:?}]", other),
                }
            ),
            2 => print!(
                "step: {} ",
                match key.read(buf)? {
                    ProtoValue::Varint(n) => format!("{:?}", n as i64),
                    other => format!("unexpected[{:?}]", other),
                }
            ),
            3 => print!(
                "file_version: {} ",
                match key.read(buf)? {
                    ProtoValue::LengthDelimited(payload) => {
                        format!("{:?}", String::from_utf8_lossy(payload))
                    }
                    other => format!("unexpected[{:?}]", other),
                }
            ),
            5 => match key.read(buf)? {
                ProtoValue::LengthDelimited(msg) => {
                    print!("summary {{ ");
                    parse_summary_proto(msg);
                    print!("}} ");
                }
                other => print!("summary {{ unexpected[{:?}] }}", other),
            },
            n => {
                print!("field{}[ignored] ", n);
                key.skip(buf)?;
            }
        };
        Some(())
    }
}

fn parse_summary_proto(message: &[u8]) -> Option<()> {
    let mut buf: &[u8] = &message[..];
    while let Some(()) = parse_summary_field(&mut buf) {}
    return Some(());

    fn parse_summary_field(buf: &mut &[u8]) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            1 => print!(
                "value: {} ",
                match key.read(buf)? {
                    ProtoValue::LengthDelimited(payload) => {
                        format!("[blob of length {}]", payload.len())
                    }
                    other => format!("unexpected[{:?}]", other),
                }
            ),
            n => {
                print!("field{}[ignored] ", n);
                key.skip(buf)?;
            }
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
