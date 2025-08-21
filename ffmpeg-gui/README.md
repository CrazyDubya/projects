# FFmpeg GUI Wrapper

A fantastic but deep GUI wrapper for the powerful multimedia tool, ffmpeg.

## Project Goals

The main goal of this project is to provide a user-friendly graphical interface for ffmpeg that is both accessible to beginners and powerful enough for advanced users. It aims to expose the vast functionality of ffmpeg in an intuitive and organized manner.

## Technology Stack

*   **Language:** Python 3
*   **GUI Framework:** PyQt6 - Chosen for its comprehensive set of modern widgets, excellent documentation, and strong community support. It allows for the creation of complex and professional-looking interfaces.

## High-Level Features

This is a non-exhaustive list of features planned for the application:

*   **Intuitive Media Conversion:** A simple interface for converting media files between various formats with common presets.
*   **Advanced Conversion Options:** Access to detailed codec settings, bitrate control, and other advanced ffmpeg parameters.
*   **Video Trimming and Splitting:** A visual timeline for selecting parts of a video to cut or split into multiple segments.
*   **Audio Extraction and Manipulation:** Tools to extract audio from video files, convert audio formats, and adjust volume.
*   **Subtitle Management:** Ability to add, remove, or convert subtitle tracks.
*   **Real-time Command Preview:** See the generated ffmpeg command before executing it.
*   **Batch Processing:** Queue up multiple files or tasks to be processed sequentially.
*   **Filter Graph Editor (The "Deep" part):** A node-based editor to visually create complex filter chains. This will allow users to chain together filters for resizing, cropping, rotating, adding watermarks, color correction, etc., in a powerful and flexible way.
*   **Preset Management:** Save and load custom conversion settings and filter chains as presets.
