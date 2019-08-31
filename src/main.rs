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
    while let Some(event) = read_event(&mut file)? {
        println!("block: {:?}", event);
    }
    Ok(())
}

/// Returns the read TF record, or `None` if at EOF.
fn read_event(f: &mut File) -> io::Result<Option<Vec<u8>>> {
    // From TensorFlow `record_writer.cc` comments:
    // Format of a single record:
    //  uint64    length
    //  uint32    masked crc of length
    //  byte      data[length]
    //  uint32    masked crc of data
    //
    // For now, just read 8-byte blocks.
    let mut buf = [0 as u8; 8];
    match f.read_exact(&mut buf) {
        Ok(()) => (),
        Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        e => e?,
    };
    Ok(Some(buf.to_vec()))
}
