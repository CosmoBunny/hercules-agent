//! Hercules Agent TUI binary.

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
use ratatui::{Terminal, backend::CrosstermBackend};
use std::{
    error::Error,
    io,
    sync::atomic::{AtomicBool, Ordering},
};

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

    let mut app = App::new();
    let res = run_app(&mut terminal, &mut app).await;

    if let Ok(mut g) = app.is_generating.lock() {
        *g = false;
    }
    cleanup_engines();
    restore_terminal(&mut terminal);

    if let Err(err) = res {
        eprintln!("{:?}", err);
    }

    std::process::exit(0);
}

async fn run_app(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> io::Result<()> {
    loop {
        if REQUEST_QUIT.load(Ordering::SeqCst) {
            app.should_quit = true;
            if let Ok(mut g) = app.is_generating.lock() {
                *g = false;
            }
            cleanup_engines();
            return Ok(());
        }
        terminal.draw(|f| app.draw(f))?;
        app.handle_events().await?;
        if app.should_quit {
            if let Ok(mut g) = app.is_generating.lock() {
                *g = false;
            }
            cleanup_engines();
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
