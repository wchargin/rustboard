use byteorder::{LittleEndian, ReadBytesExt};
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
    fn skip(it: &mut &[u8], n: u64) {
        *it = &it[n as usize..];
    }
    while let Some(key) = read_varu64(&mut buf) {
        let wire_type = key & 0b111;
        let field_number = (key & !0b111) >> 3;
        match wire_type {
            0 => {
                // single varint
                read_varu64(&mut buf);
            }
            2 => {
                // length-delimited
                let len = read_varu64(&mut buf).unwrap();
                skip(&mut buf, len);
            }
            1 => skip(&mut buf, 8), // fixed 64-bit
            5 => skip(&mut buf, 4), // fixed 32-bit
            n => unimplemented!("wire type format {}", n),
        }
        print!("f#{}[wt={}] ", field_number, wire_type);
    }
    println!("<end>");
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
