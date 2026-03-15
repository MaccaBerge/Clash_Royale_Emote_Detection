class State:
    startup_state_string = "startup"
    menu_state_string = "menu"
    detection_state_string = "detection"
    error_state_string = "error"


class Path:
    main_font_path = "assets/fonts/Inter.ttc"
    asset_folder_path = "assets"
    prediction_model_path = "models/model.joblib"

    app_help_dir = {
        "darwin": "~/Library/Application Support",
        "windows": "~/AppData/Roaming",
        "linux": "~/.local/share",
    }


class Logger:
    app_logger_name = "app"
    app_logger_format = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    app_logger_date_format = "%d-%m-%YT%H:%M:%S"

    log_file_name = "program.log"
    log_file_max_bytes = 1_000_000
    log_file_backup_count = 3
    log_file_encoding = "utf-8"


class ErrorScreen:
    default_error_message = "Some error occured"
    background_color = (238, 75, 43)
    text_color = (255, 255, 255)


class General:
    program_name = "clash_royale_emote_detection"
    pygame_mixer_num_channels = 8
    prediction_model_num_features = 13
    background_color = (255, 255, 255)


class Constants:
    state = State()
    path = Path()
    general = General()
    logger = Logger()
    error_screen = ErrorScreen()
