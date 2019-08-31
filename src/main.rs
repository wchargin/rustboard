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
            Ok(block) => println!(
                "read block of length {}: {:?}",
                block.len(),
                String::from_utf8_lossy(&block)
            ),
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
