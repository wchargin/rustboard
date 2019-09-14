use byteorder::{ByteOrder, LittleEndian, ReadBytesExt};
use log::{error, info, warn};
use rand_chacha::ChaChaRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use walkdir::WalkDir;

const DEFAULT_RESERVOIR_SIZE: usize = 1000;

fn main() {
    use clap::Arg;
    let matches = clap::App::new("rustboard")
        .arg(
            Arg::with_name("logdir")
                .long("logdir")
                .value_name("LOGDIR")
                .help("Directory from which to load data; will be traversed recursively")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("inspect")
                .long("inspect")
                .help("Print statistics about loaded data rather than starting a server"),
        )
        .arg(
            Arg::with_name("verbose")
                .long("verbose")
                .help("Print more information."),
        )
        .arg(
            Arg::with_name("downsample")
                .long("downsample")
                .help("Downsample to this number of points per time series")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("host")
                .long("host")
                .help("Host to bind to: e.g., 'localhost' or '0.0.0.0'")
                .takes_value(true)
                .default_value("localhost"),
        )
        .arg(
            Arg::with_name("port")
                .long("port")
                .help("Port to bind to: e.g., 6006")
                .default_value("6006"),
        )
        .get_matches();

    env_logger::from_env(env_logger::Env::default().default_filter_or(
        if matches.is_present("verbose") {
            "info"
        } else {
            "warn"
        },
    ))
    .default_format_timestamp_nanos(true)
    .init();
    let logdir = matches.value_of("logdir").unwrap();
    let inspect = matches.is_present("inspect");
    let reservoir_size: usize = matches
        .value_of("downsample")
        .map(|x| {
            x.parse::<usize>()
                .expect("--downsample must be a non-negative integer")
        })
        .unwrap_or(DEFAULT_RESERVOIR_SIZE);
    let host = matches.value_of("host").expect("has default value");
    let port = matches
        .value_of("port")
        .expect("has default value")
        .parse::<u16>()
        .expect("--port must be a u16");

    info!("Starting loading");

    let mut events_files_by_run = HashMap::<RunId, Vec<PathBuf>>::new();
    for dirent in WalkDir::new(logdir)
        .into_iter()
        .filter_map(|result| result.ok())
    {
        if !dirent.file_type().is_file() {
            continue;
        }
        if !dirent.file_name().to_string_lossy().contains("tfevents") {
            continue;
        }
        let run = dirent
            .path()
            .parent()
            .map(|x| {
                x.strip_prefix(logdir)
                    .unwrap_or(x)
                    .to_string_lossy()
                    .into_owned()
            })
            .unwrap_or_else(|| ".".to_string());
        events_files_by_run
            .entry(RunId(run))
            .or_default()
            .push(dirent.path().to_owned());
    }

    info!(
        "Discovered {} runs ({:?} event files)",
        events_files_by_run.len(),
        events_files_by_run.values().map(|x| x.len()).sum::<usize>()
    );

    let multiplexer = Arc::new(Mutex::new(ScalarsMultiplexer::new()));
    let mut threads = Vec::new();
    for (run, events_files) in events_files_by_run.into_iter() {
        let multiplexer = Arc::clone(&multiplexer);
        threads.push(std::thread::spawn(move || {
            let mut accumulator = ScalarsAccumulator::from_reservoir_size(reservoir_size);
            for events_file in events_files {
                read_events(events_file, &mut accumulator).unwrap_or_else(|e| {
                    error!("{:?}", e);
                });
            }
            for (tag, time_series) in &accumulator.time_series {
                if time_series.skipped_points > 0 {
                    warn!(
                        "Skipped {} step-decreasing point(s) in time series {:?} of run {:?}",
                        time_series.skipped_points, tag.0, run.0,
                    );
                }
            }
            let mut multiplexer = multiplexer
                .lock()
                .expect("mutex poisoned; sibling thread panicked");
            multiplexer.runs.insert(run, accumulator);
        }));
    }
    for thread in threads.into_iter() {
        thread.join().expect("child thread panicked");
    }

    let multiplexer: ScalarsMultiplexer = Arc::try_unwrap(multiplexer)
        .map_err(|_| ())
        .expect("lingering reference")
        .into_inner()
        .expect("mutex poisoned; child thread panicked");

    info!("Done loading");

    if inspect {
        info!("Read data for {} run(s)", multiplexer.runs.len());
        for (run, accumulator) in &multiplexer.runs {
            info!("* {}", run.0);
            for (tag, ts) in &accumulator.time_series {
                info!(
                    "  - {} ({} points; max step {})",
                    tag.0,
                    ts.points.seen,
                    ts.max_step
                        .map(|x| format!("{:?}", x))
                        .unwrap_or_else(|| "<no data>".to_string())
                );
            }
        }
    } else {
        server::AppState {
            logdir: logdir.to_string(),
            multiplexer,
            host: host.to_string(),
            port,
        }
        .serve();
    }
}

mod server {
    use super::*;
    use actix_files::NamedFile;
    use actix_web::{web, App, HttpServer, Responder};

    pub struct AppState {
        pub logdir: String,
        pub multiplexer: ScalarsMultiplexer,
        pub host: String,
        pub port: u16,
    }
    type AppData = Arc<AppState>;

    fn index() -> impl Responder {
        NamedFile::open("index.html")
    }

    #[derive(Serialize)]
    struct PluginStatus {
        disable_reload: bool,
        enabled: bool,
        loading_mechanism: LoadingMechanism,
        remove_dom: bool,
        tab_name: &'static str,
    }

    #[derive(Serialize)]
    #[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
    enum LoadingMechanism {
        CustomElement {
            element_name: &'static str,
        },
        #[allow(dead_code)]
        Iframe {
            es_module_path: &'static str,
        },
        #[allow(dead_code)]
        None,
    }

    #[derive(Serialize)]
    struct PluginsListingResponse(HashMap<&'static str, PluginStatus>);

    fn data_plugins_listing(data: web::Data<AppData>) -> impl Responder {
        let have_scalars = data
            .multiplexer
            .runs
            .values()
            .flat_map(|accumulator| accumulator.time_series.values())
            .any(|ts| !ts.points.items.is_empty());
        let mut res = PluginsListingResponse(HashMap::new());
        res.0.insert(
            "scalars",
            PluginStatus {
                disable_reload: false,
                enabled: have_scalars,
                loading_mechanism: LoadingMechanism::CustomElement {
                    element_name: "tf-scalar-dashboard",
                },
                tab_name: "scalars",
                remove_dom: false,
            },
        );
        web::Json(res)
    }

    fn data_runs(data: web::Data<AppData>) -> impl Responder {
        let mut runs = data
            .multiplexer
            .runs
            .iter()
            .map(|(run_id, accumulator)| (run_id.clone(), accumulator.first_event_timestamp))
            .collect::<Vec<_>>();
        runs.sort_by(|&(_, t1), &(_, t2)| t1.partial_cmp(&t2).unwrap());
        web::Json(
            runs.into_iter()
                .map(|(run_id, _timestamp)| run_id)
                .collect::<Vec<_>>(),
        )
    }

    fn data_logdir(data: web::Data<AppData>) -> impl Responder {
        web::Json(data.logdir.clone())
    }

    #[derive(Serialize)]
    struct EnvironmentResponse<'a> {
        window_title: &'a str,
        mode: &'a str,
        data_location: String,
    }

    fn data_environment(data: web::Data<AppData>) -> impl Responder {
        web::Json(EnvironmentResponse {
            window_title: "TensorBoard",
            mode: "logdir",
            data_location: data.logdir.clone(),
        })
    }

    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct TagInfo {
        // TODO(wchargin): Avoid copies here and throughout this handler.
        display_name: String,
        description: String,
    }

    impl TagInfo {
        fn new() -> TagInfo {
            TagInfo {
                display_name: String::new(),
                description: String::new(),
            }
        }
    }

    #[derive(Serialize)]
    struct TagsResponse(HashMap<RunId, HashMap<TagId, TagInfo>>);

    fn data_plugin_scalars_tags(data: web::Data<AppData>) -> impl Responder {
        let mut result: TagsResponse = TagsResponse(HashMap::new());
        for (run, accumulator) in &data.multiplexer.runs {
            let run_info = result.0.entry(run.clone()).or_default();
            for tag in accumulator.time_series.keys() {
                run_info.insert(tag.clone(), TagInfo::new());
            }
        }
        web::Json(result)
    }

    #[derive(Deserialize)]
    struct ScalarsRequest {
        run: String,
        tag: String,
    }

    #[derive(Serialize)]
    struct ScalarsResponse(Vec<(f64, i64, f32)>);

    fn data_plugin_scalars_scalars(
        data: web::Data<AppData>,
        query: web::Query<ScalarsRequest>,
    ) -> Result<web::Json<ScalarsResponse>, actix_web::error::Error> {
        use actix_web::error::ErrorBadRequest;
        let time_series = data
            .multiplexer
            .runs
            .get(&query.run as &str)
            .and_then(|acc| acc.time_series.get(&query.tag as &str))
            .ok_or_else(|| ErrorBadRequest("Invalid run/tag"))?;
        let result = time_series
            .points
            .items
            .iter()
            .map(|pt| (pt.wall_time, pt.step, pt.value))
            .collect::<Vec<_>>();
        Ok(web::Json(ScalarsResponse(result)))
    }

    impl AppState {
        pub fn serve(self) {
            let address = format!("{}:{}", &self.host, self.port);
            let shared_state = Arc::new(self);
            let server = HttpServer::new(move || {
                App::new()
                    .route("/", web::get().to(index))
                    .route("/index", web::get().to(index))
                    .route("/data/runs", web::get().to(data_runs))
                    .route("/data/logdir", web::get().to(data_logdir))
                    .route("/data/environment", web::get().to(data_environment))
                    .route("/data/plugins_listing", web::get().to(data_plugins_listing))
                    .route(
                        "/data/plugin/scalars/tags",
                        web::get().to(data_plugin_scalars_tags),
                    )
                    .route(
                        "/data/plugin/scalars/scalars",
                        web::get().to(data_plugin_scalars_scalars),
                    )
                    .data(shared_state.clone())
                    .wrap(actix_web::middleware::Logger::default())
            })
            .bind(&address)
            .expect("Failed to bind");
            println!("Started web server at http://{}", &address);
            server.run().expect("Failed to run");
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Clone)]
struct RunId(String);

impl std::borrow::Borrow<str> for RunId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Clone)]
struct TagId(String);

impl std::borrow::Borrow<str> for TagId {
    fn borrow(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, PartialEq)]
struct ScalarPoint {
    /// Step index: unique non-negative key for this datum within its time series.
    step: i64,
    /// Wall time that the event was recorded, as seconds since epoch.
    wall_time: f64,
    /// Scalar value.
    value: f32,
}

pub struct ScalarsMultiplexer {
    runs: HashMap<RunId, ScalarsAccumulator>,
}

impl ScalarsMultiplexer {
    fn new() -> Self {
        ScalarsMultiplexer {
            runs: HashMap::new(),
        }
    }
}

struct ScalarsAccumulator {
    reservoir_size: usize,
    first_event_timestamp: Option<f64>,
    known_tags: HashMap<TagId, bool>, // is scalars?
    time_series: HashMap<TagId, TimeSeries>,
}

struct TimeSeries {
    max_step: Option<i64>,
    skipped_points: usize,
    points: Reservoir<ScalarPoint>,
}

impl TimeSeries {
    fn add(&mut self, pt: ScalarPoint) {
        match self.max_step {
            None => self.max_step = Some(pt.step),
            Some(ref mut max_step) if *max_step < pt.step => (*max_step = pt.step),
            _ => {
                self.skipped_points += 1;
                return;
            }
        };
        self.points.add(pt)
    }
}

impl ScalarsAccumulator {
    fn from_reservoir_size(reservoir_size: usize) -> Self {
        ScalarsAccumulator {
            reservoir_size,
            first_event_timestamp: None,
            known_tags: HashMap::new(),
            time_series: HashMap::new(),
        }
    }
    fn series(&mut self, tag: TagId) -> &mut TimeSeries {
        let size = self.reservoir_size;
        self.time_series.entry(tag).or_insert_with(|| TimeSeries {
            max_step: None,
            skipped_points: 0,
            points: Reservoir::new(size),
        })
    }
}

#[derive(Debug)]
pub struct Reservoir<T> {
    capacity: usize,
    pub items: Vec<T>,
    rng: ChaChaRng,
    seen: usize,
}

impl<T> Reservoir<T> {
    fn new(capacity: usize) -> Self {
        use rand::SeedableRng;
        Reservoir {
            capacity,
            items: Vec::with_capacity(capacity),
            rng: rand_chacha::ChaCha20Rng::from_seed(std::default::Default::default()),
            seen: 0,
        }
    }

    fn add(&mut self, t: T) {
        use rand::distributions::{Distribution, Uniform};
        self.seen += 1;
        if self.items.len() < self.capacity {
            self.items.push(t);
        } else if Uniform::from(0..self.seen).sample(&mut self.rng) < self.capacity {
            self.items
                .remove(Uniform::from(0..self.capacity).sample(&mut self.rng));
            self.items.push(t);
        }
    }
}

fn read_events<P: AsRef<Path>>(
    filename: P,
    accumulator: &mut ScalarsAccumulator,
) -> io::Result<()> {
    let file = File::open(filename)?;
    let mut reader = io::BufReader::new(file);
    loop {
        match read_event(&mut reader) {
            Ok(block) => parse_event_proto(&block, accumulator),
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(())
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

fn parse_event_proto(event: &[u8], accumulator: &mut ScalarsAccumulator) {
    let mut buf: &[u8] = &event[..];
    let mut wall_time: f64 = 0.0;
    let mut step: i64 = 0;
    let mut tag_values: Vec<TagValue> = Vec::new();
    while let Some(()) = parse_event_field(
        &mut buf,
        &mut wall_time,
        &mut step,
        &mut tag_values,
        accumulator,
    ) {}
    accumulator.first_event_timestamp.get_or_insert(wall_time);
    for tag_value in tag_values.into_iter() {
        accumulator.series(tag_value.tag).add(ScalarPoint {
            step,
            wall_time,
            value: tag_value.value,
        })
    }

    // Relevant fields on `Event`:
    //   double wall_time = 1;
    //   inter step = 2;
    //   string file_version = 3;
    //   Summary summary = 5;
    fn parse_event_field(
        buf: &mut &[u8],
        wall_time: &mut f64,
        step: &mut i64,
        tag_values: &mut Vec<TagValue>,
        accumulator: &mut ScalarsAccumulator,
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
                    if let Some(tvs) = parse_summary_proto(summary_msg, accumulator) {
                        tag_values.extend(tvs.into_iter());
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

fn parse_summary_proto(
    message: &[u8],
    accumulator: &mut ScalarsAccumulator,
) -> Option<Vec<TagValue>> {
    let mut buf: &[u8] = &message[..];
    let mut result = Vec::new();
    while let Some(()) = parse_summary_field(&mut buf, &mut result, accumulator) {}
    return Some(result);

    // Relevant fields on `Summary`:
    //   repeated Value value = 1;
    fn parse_summary_field(
        buf: &mut &[u8],
        result: &mut Vec<TagValue>,
        accumulator: &mut ScalarsAccumulator,
    ) -> Option<()> {
        let key = ProtoKey::new(read_varu64(buf)?);
        match key.field_number {
            1 => {
                if let Some(value_msg) = key.read_length_delimited(buf) {
                    result.extend(parse_value_proto(value_msg, accumulator).into_iter())
                }
            }
            _ => key.skip(buf)?,
        };
        Some(())
    }
}

fn parse_value_proto(message: &[u8], accumulator: &mut ScalarsAccumulator) -> Option<TagValue> {
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
    let (tag, value, from_scalars_plugin) = match result {
        PartialTagValue {
            tag: Some(tag),
            value: Some(value),
            from_scalars_plugin,
        } => (tag, value, from_scalars_plugin),
        _ => return None,
    };
    let is_scalars_tag = match accumulator.known_tags.get(&tag) {
        Some(v) => *v,
        None => {
            accumulator
                .known_tags
                .insert(tag.clone(), from_scalars_plugin);
            from_scalars_plugin
        }
    };
    return if is_scalars_tag {
        Some(TagValue { tag, value })
    } else {
        None
    };

    // Relevant fields on `Value`:
    //   string tag = 1;
    //   SummaryMetadata metadata = 9;
    //   float simple_value = 2;
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
            2 => {
                if let Some(simple_value_bits) = key.read_fixed32(buf) {
                    result.value = Some(f32::from_bits(simple_value_bits));
                    result.from_scalars_plugin = true;
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
                    if let Some(v) = parse_plugin_data_proto(plugin_data_msg) {
                        *plugin_name = Some(v);
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
        result |= u64::from(byte & 0x7F) << (i * 7);
        if byte & 0x80 == 0 {
            return Some(result);
        }
    }
    None // took too many bytes, still wasn't done
}
