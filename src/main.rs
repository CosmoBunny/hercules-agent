//! Hercules Agent TUI binary.

use clap::Parser;
use crossterm::{
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use hercules_agent::app::App;
use hercules_agent::llama;
use hercules_agent::session;
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{
    error::Error,
    io,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Parser, Debug)]
#[command(
    name = "hercules",
    version = env!("CARGO_PKG_VERSION"),
    about = "Hercules Agent — local AI coding TUI",
    disable_version_flag = true
)]
pub struct Cli {
    /// Print version information
    #[arg(short = 'v', long = "version", action = clap::ArgAction::Version)]
    pub version: (),

    /// Show current directory's session ID and exit
    #[arg(short = 's', long = "session")]
    pub session: bool,

    /// Continue existing session for current directory, or specify a session ID
    #[arg(
        short = 'c',
        long = "continue",
        num_args = 0..=1,
        value_name = "SESSION_ID"
    )]
    pub continue_session: Option<Option<String>>,

    /// Clear all unlocked sessions for current directory and exit
    #[arg(long = "clear-session")]
    pub clear_session: bool,

    /// Clear all unlocked sessions across all directories and exit
    #[arg(long = "clear-all-session")]
    pub clear_all_session: bool,
}

/// Set on SIGTERM only — request clean app quit (kill servers).
/// SIGINT / Ctrl+C is **not** handled here so the TUI can interrupt generation only.
static REQUEST_QUIT: AtomicBool = AtomicBool::new(false);

fn cleanup_engines() {
    llama::server::shutdown_managed_server();
    llama::libinfer::shutdown_warm_lib_engine();
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) {
    let _ = disable_raw_mode();
    let _ = execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        DisableBracketedPaste,
        LeaveAlternateScreen,
        PopKeyboardEnhancementFlags
    );
    let _ = terminal.show_cursor();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let cwd = std::env::current_dir()?;

    // If --session / -s: print session ID for current directory and exit immediately.
    if cli.session {
        let sid = session::session_id_for_dir(&cwd);
        println!("{}", sid);
        return Ok(());
    }

    // If --clear-session: clear all unlocked sessions for current directory and exit.
    if cli.clear_session {
        let (cleared, skipped) = session::clear_session_for_dir(&cwd);
        if skipped > 0 {
            println!(
                "Cleared {} session(s) for current directory ({} active/locked session(s) skipped).",
                cleared, skipped
            );
        } else {
            println!("Cleared {} session(s) for current directory.", cleared);
        }
        return Ok(());
    }

    // If --clear-all-session: clear all unlocked sessions across all directories and exit.
    if cli.clear_all_session {
        let (cleared, skipped) = session::clear_all_sessions();
        if skipped > 0 {
            println!(
                "Cleared {} session(s) across all directories ({} active/locked session(s) skipped).",
                cleared, skipped
            );
        } else {
            println!("Cleared {} session(s) across all directories.", cleared);
        }
        return Ok(());
    }

    unsafe {
        std::env::set_var("RUST_LOG", "error");
        std::env::set_var("WGPU_LOG", "error");
    }

    // SIGTERM (e.g. kill): quit cleanly. Do **not** hook SIGINT — Ctrl+C is
    // handled in the TUI as "interrupt agent" without killing llama-server.
    let _ = sigterm_set_handler(|| {
        REQUEST_QUIT.store(true, Ordering::SeqCst);
    });

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    let _ = execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste,
        // Key release needed so Esc hold cancels when released early
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_EVENT_TYPES
        )
    );
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let is_continue = cli.continue_session.is_some();
    let target_sid = match cli.continue_session {
        Some(Some(ref explicit_id)) if !explicit_id.trim().is_empty() => explicit_id.clone(),
        Some(_) => {
            if let Some(latest) = session::latest_session_for_dir(&cwd) {
                latest.session_id
            } else {
                session::new_session_id_for_dir(&cwd)
            }
        }
        None => session::new_session_id_for_dir(&cwd),
    };

    let _lock_guard = match session::acquire_session_lock(&target_sid) {
        Ok(guard) => Some(guard),
        Err(err) => {
            eprintln!("Warning: {}", err);
            None
        }
    };

    let mut app = if is_continue {
        let s = session::load_or_create_session(&target_sid, &cwd);
        App::with_session(s)
    } else {
        let mut app = App::new();
        app.session_id = Some(target_sid);
        app
    };

    let res = run_app(&mut terminal, &mut app).await;

    if let Ok(mut g) = app.is_generating.lock() {
        *g = false;
    }

    app.save_current_session();

    // Release lock file explicitly before _exit
    drop(_lock_guard);
    if let Some(ref sid) = app.session_id {
        session::release_session_lock(sid);
    }

    // Restore terminal first so the user's shell is usable immediately.
    restore_terminal(&mut terminal);

    if let Err(err) = res {
        eprintln!("{:?}", err);
    }

    // Use _exit() instead of process::exit() to bypass llama.cpp's atexit
    // handlers. Those handlers call llama_backend_free / context/model free
    // which can block for several seconds freeing a 1-2GB model on slow
    // hardware, causing the process to hang visibly at 99% exit.
    // The OS reclaims all memory immediately on _exit — no cleanup needed.
    unsafe { libc::_exit(0) };
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> io::Result<()> {
    // Always draw once at startup so the user sees the UI immediately.
    terminal.draw(|f| app.draw(f))?;
    loop {
        if REQUEST_QUIT.load(Ordering::SeqCst) {
            app.should_quit = true;
            if let Ok(mut g) = app.is_generating.lock() {
                *g = false;
            }
            // Spawn cleanup off the event loop — don't block here.
            std::thread::spawn(cleanup_engines);
            return Ok(());
        }
        let needs_redraw = app.handle_events().await?;
        if needs_redraw {
            terminal.draw(|f| app.draw(f))?;
        }
        if app.should_quit {
            if let Ok(mut g) = app.is_generating.lock() {
                *g = false;
            }
            // Spawn cleanup so we don't block exit on GEN_LOCK held by a generate thread.
            std::thread::spawn(cleanup_engines);
            return Ok(());
        }
    }
}

fn sigterm_set_handler<F>(handler: F) -> Result<(), String>
where
    F: Fn() + Send + Sync + 'static,
{
    use std::sync::OnceLock;
    static HANDLER: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();
    let _ = HANDLER.set(Box::new(handler));

    #[cfg(unix)]
    {
        unsafe extern "C" fn trampoline(_: i32) {
            if let Some(h) = HANDLER.get() {
                h();
            }
        }
        unsafe {
            let h = trampoline as *const () as libc::sighandler_t;
            libc::signal(libc::SIGTERM, h);
            // Leave SIGINT default / TUI — app maps Ctrl+C to interrupt only
        }
    }
    Ok(())
}
