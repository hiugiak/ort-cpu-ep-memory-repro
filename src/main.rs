use std::{env, path::PathBuf, time::Instant};

use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};

const INPUT_LEN: usize = 1 * 3 * 640 * 640;
const INPUT_NAME: &str = "images";

fn main() -> ort::Result<()> {
    let config = Config::from_args();

    ort::init_from(&config.ort_library_path)?.commit();

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(config.intra_threads)?
        .with_execution_providers([ep::CPU::default().build().error_on_failure()])?
        .commit_from_file(&config.model_path)?;

    println!("ort_library={}", config.ort_library_path.display());
    println!("model={}", config.model_path.display());
    println!("input=noise: fresh random tensor per iteration");
    println!("intra_threads={}", config.intra_threads);
    println!("print_every={}", config.print_every);
    println!("rss_start_mb={:.1}", rss_mb());

    let started = Instant::now();

    for iteration in 1_u64.. {
        let input = generate_noise_tensor(iteration);
        run_inference(&mut session, &input)?;

        if iteration % config.print_every == 0 {
            println!(
                "iter={iteration}, elapsed={:.1}s, rss={:.1} MB",
                started.elapsed().as_secs_f64(),
                rss_mb(),
            );
        }
    }

    Ok(())
}

struct Config {
    ort_library_path: PathBuf,
    model_path: PathBuf,
    intra_threads: usize,
    print_every: u64,
}

impl Config {
    fn from_args() -> Self {
        let mut args = env::args_os().skip(1);
        let ort_library_path = args.next().map(PathBuf::from).unwrap_or_else(|| {
            usage_and_exit("missing path to libonnxruntime.dylib");
        });
        let model_path = args.next().map(PathBuf::from).unwrap_or_else(|| {
            usage_and_exit("missing path to ONNX model");
        });

        let intra_threads = parse_optional_arg(args.next(), 1usize, "intra_threads");
        let print_every = parse_optional_arg(args.next(), 100u64, "print_every");

        Self {
            ort_library_path,
            model_path,
            intra_threads,
            print_every,
        }
    }
}

fn generate_noise_tensor(iteration: u64) -> Vec<f32> {
    let mut rng = SmallRng::seed_from_u64(iteration);
    let mut input = vec![0.0f32; INPUT_LEN];
    for value in &mut input {
        *value = rng.r#gen();
    }
    input
}

fn run_inference(session: &mut Session, input: &[f32]) -> ort::Result<()> {
    let outputs = session.run(ort::inputs! {
        INPUT_NAME => TensorRef::from_array_view((
            [1_usize, 3, 640, 640],
            input,
        ))?
    })?;

    let (_, output_value) = outputs
        .iter()
        .next()
        .expect("expected at least one model output");
    let output = output_value.try_extract_tensor::<f32>()?;
    std::hint::black_box(output);
    Ok(())
}

fn parse_optional_arg<T>(value: Option<std::ffi::OsString>, default: T, label: &str) -> T
where
    T: std::str::FromStr,
{
    let Some(value) = value else {
        return default;
    };
    value
        .to_string_lossy()
        .parse()
        .unwrap_or_else(|_| usage_and_exit(&format!("invalid {label}")))
}

fn usage_and_exit(message: &str) -> ! {
    eprintln!("{message}");
    eprintln!(
        "usage: ort-cpu-ep-memory-repro <libonnxruntime.dylib> <model.onnx> [intra_threads] [print_every]"
    );
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
fn rss_mb() -> f64 {
    let mut info = std::mem::MaybeUninit::<libc::mach_task_basic_info>::uninit();
    let mut count = (std::mem::size_of::<libc::mach_task_basic_info>()
        / std::mem::size_of::<libc::natural_t>())
        as libc::mach_msg_type_number_t;

    let result = unsafe {
        libc::task_info(
            mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast(),
            &mut count,
        )
    };

    if result != libc::KERN_SUCCESS {
        return f64::NAN;
    }

    let info = unsafe { info.assume_init() };
    info.resident_size as f64 / 1024.0 / 1024.0
}

#[cfg(target_os = "macos")]
fn mach_task_self() -> libc::mach_port_t {
    unsafe extern "C" {
        static mach_task_self_: libc::mach_port_t;
    }

    unsafe { mach_task_self_ }
}

#[cfg(not(target_os = "macos"))]
fn rss_mb() -> f64 {
    f64::NAN
}
