import os
import shutil
import subprocess
from datetime import datetime
import pygame
import threading
import time
import math


class VideoRecorder:
    def __init__(self, fps=30, temp_dir="temp_frames", logo_path=None, intro_sec=4.0, fade_sec=1.0):
        self.fps = fps
        self.temp_dir = temp_dir
        self.recording = False
        self.frame_count = 0
        self.session_id = None
        self.frame_dir = None
        self.logo_path = logo_path
        self.intro_sec = intro_sec
        self.fade_sec = fade_sec

        # Progress tracking
        self.processing = False
        self.progress_callback = None
        self.processing_thread = None

    def start_recording(self):
        """Start a new recording session."""
        if self.recording:
            return False

        # Create session ID based on timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_dir = os.path.join(self.temp_dir, self.session_id)

        # Create directory for frames
        os.makedirs(self.frame_dir, exist_ok=True)

        self.recording = True
        self.frame_count = 0
        print(f"Recording started: {self.session_id}")
        return True

    def save_frame(self, surface):
        """Save a single frame to disk."""
        if not self.recording:
            return

        # Save frame with zero-padded filename for proper ordering
        filename = os.path.join(self.frame_dir, f"frame_{self.frame_count:08d}.png")
        pygame.image.save(surface, filename)
        self.frame_count += 1

    def stop_recording(self, output_dir="simulation_recordings", keep_frames=False, progress_callback=None):
        """Stop recording and compile frames into video with progress tracking."""
        if not self.recording:
            return None

        self.recording = False
        self.progress_callback = progress_callback
        print(f"Recording stopped: {self.frame_count} frames captured")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Output video filename
        output_filename = os.path.join(output_dir, f"simulation_{self.session_id}.mp4")

        # Start processing in background thread
        self.processing = True
        self.processing_thread = threading.Thread(
            target=self._process_video_async,
            args=(output_filename, keep_frames)
        )
        self.processing_thread.start()

        return output_filename

    def _process_video_async(self, output_filename, keep_frames):
        """Process video compilation in background thread with progress updates."""
        try:
            success = self.compile_video_with_progress(output_filename)
            if success:
                self._report_progress("Adding logo intro...", 90)
                self._prepend_logo_intro(output_filename)
                self._report_progress("Finalizing video...", 95)

            # Clean up temporary frames unless requested to keep them
            if not keep_frames and success:
                self._report_progress("Cleaning up temporary files...", 98)
                shutil.rmtree(self.frame_dir)
                print(f"Temporary frames deleted")

            self._report_progress("Video processing complete!", 100)

            if success:
                print(f"Video successfully created: {output_filename}")
            else:
                print(f"Video processing failed")

        except Exception as e:
            print(f"Error during video processing: {e}")
        finally:
            self.processing = False

    def _report_progress(self, message, percentage):
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(message, percentage)
        print(f"Progress: {percentage}% - {message}")

    def compile_video_with_progress(self, output_filename):
        """Compile frames into video with progress tracking."""
        input_pattern = os.path.join(self.frame_dir, "frame_%08d.png")

        # Estimate total frames for progress calculation
        total_frames = self.frame_count

        self._report_progress("Starting video compilation...", 5)

        # For very long videos, process in batches to avoid memory issues
        if total_frames > 10000:  # More than ~5.5 minutes at 30fps
            return self._compile_video_batched(output_filename, total_frames)
        else:
            return self._compile_video_single(output_filename, total_frames)

    def _compile_video_single(self, output_filename, total_frames):
        """Compile video in single pass with progress monitoring."""
        input_pattern = os.path.join(self.frame_dir, "frame_%08d.png")

        # FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-r', str(self.fps),
            '-i', input_pattern,
            '-vf', f'fade=t=in:st=0:d={self.fade_sec}',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '0',
            '-preset', 'ultrafast',
            '-r', str(self.fps),
            '-progress', 'pipe:1',  # Enable progress output
            output_filename
        ]

        try:
            self._report_progress("Compiling video frames...", 10)

            # Run ffmpeg with progress monitoring
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, bufsize=1, universal_newlines=True)

            # Monitor progress
            last_progress = 10
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break

                if output.startswith('frame='):
                    try:
                        frame_num = int(output.split('=')[1].strip())
                        progress = min(85, 10 + int(75 * frame_num / total_frames))
                        if progress > last_progress:
                            self._report_progress(f"Processing frame {frame_num}/{total_frames}...", progress)
                            last_progress = progress
                    except (ValueError, IndexError):
                        pass

            # Wait for completion
            process.wait()

            if process.returncode == 0:
                self._report_progress("Base video compiled successfully", 85)
                return True
            else:
                stderr = process.stderr.read()
                print(f"Error creating video: {stderr}")
                return False

        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg to create videos.")
            print("Frames are saved in:", self.frame_dir)
            return False
        except Exception as e:
            print(f"Error creating video: {e}")
            return False

    def _compile_video_batched(self, output_filename, total_frames):
        """Compile very long videos in batches to avoid memory issues."""
        batch_size = 5000  # Process 5000 frames at a time
        num_batches = math.ceil(total_frames / batch_size)

        self._report_progress(f"Processing {total_frames} frames in {num_batches} batches...", 10)

        batch_files = []

        try:
            # Process each batch
            for batch_idx in range(num_batches):
                start_frame = batch_idx * batch_size
                end_frame = min((batch_idx + 1) * batch_size - 1, total_frames - 1)

                batch_output = os.path.join(self.frame_dir, f"batch_{batch_idx:03d}.mp4")
                batch_files.append(batch_output)

                # Create input pattern for this batch
                input_pattern = os.path.join(self.frame_dir, f"frame_%08d.png")

                cmd = [
                    'ffmpeg', '-y',
                    '-r', str(self.fps),
                    '-start_number', str(start_frame),
                    '-i', input_pattern,
                    '-frames:v', str(end_frame - start_frame + 1),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '0',
                    '-preset', 'ultrafast',
                    '-r', str(self.fps),
                    batch_output
                ]

                batch_progress = 10 + int(60 * batch_idx / num_batches)
                self._report_progress(
                    f"Processing batch {batch_idx + 1}/{num_batches} (frames {start_frame}-{end_frame})...",
                    batch_progress)

                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error creating batch {batch_idx}: {result.stderr}")
                    return False

            # Concatenate all batches
            self._report_progress("Combining batches...", 70)
            concat_list = os.path.join(self.frame_dir, "concat_list.txt")
            with open(concat_list, 'w') as f:
                for batch_file in batch_files:
                    f.write(f"file '{os.path.abspath(batch_file)}'\n")

            # Apply fade effect to the final concatenated video
            temp_output = os.path.join(self.frame_dir, "temp_concat.mp4")
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0', '-i', concat_list,
                '-vf', f'fade=t=in:st=0:d={self.fade_sec}',
                '-c:v', 'libx264',
                '-crf', '0',
                '-preset', 'ultrafast',
                temp_output
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error concatenating batches: {result.stderr}")
                return False

            # Move final file to output location
            os.rename(temp_output, output_filename)

            # Clean up batch files
            for batch_file in batch_files:
                try:
                    os.remove(batch_file)
                except:
                    pass
            try:
                os.remove(concat_list)
            except:
                pass

            self._report_progress("Batched video compilation complete", 85)
            return True

        except Exception as e:
            print(f"Error in batched compilation: {e}")
            return False

    def cancel_recording(self):
        """Cancel current recording and clean up frames without creating video."""
        if not self.recording:
            return False

        self.recording = False
        print(f"Recording cancelled: {self.frame_count} frames discarded")

        # Clean up temporary frames
        if os.path.exists(self.frame_dir):
            shutil.rmtree(self.frame_dir)
            print(f"Temporary frames deleted")

        # Reset state
        self.frame_count = 0
        self.session_id = None
        self.frame_dir = None

        return True

    def toggle_recording(self, output_dir="simulation_recordings", keep_frames=False, progress_callback=None):
        """Toggle recording on/off."""
        if self.recording:
            return self.stop_recording(output_dir, keep_frames, progress_callback)
        else:
            self.start_recording()
            return None

    def is_recording(self):
        """Check if currently recording."""
        return self.recording

    def is_processing(self):
        """Check if currently processing video."""
        return self.processing

    def get_status(self):
        """Get current recording status."""
        if self.recording:
            return f"Recording: {self.frame_count} frames"
        elif self.processing:
            return "Processing video..."
        else:
            return "Not recording"

    def _prepend_logo_intro(self, main_video):
        """Build a fading-logo clip and concatenate it in front of main_video."""
        if not self.logo_path or not os.path.isfile(self.logo_path):
            print(f"Logo path not found or invalid: {self.logo_path}")
            return

        print(f"Adding logo intro to video: {main_video}")

        intro_mp4 = os.path.join(self.frame_dir, "intro.mp4")
        concat_txt = os.path.join(self.frame_dir, "concat.txt")
        final_video = os.path.join(os.path.dirname(main_video), f"temp_{os.path.basename(main_video)}")

        # Derive the simulation's width/height from the very first PNG we wrote
        first_png = os.path.join(self.frame_dir, "frame_00000000.png")
        if not os.path.exists(first_png):
            print(f"First frame not found: {first_png}")
            return

        try:
            w, h = pygame.image.load(first_png).get_size()
            print(f"Video dimensions: {w}x{h}")
        except Exception as e:
            print(f"Error getting video dimensions: {e}")
            return

        # Calculate 3/4 of screen dimensions for logo
        logo_w = int(w * 0.75)
        logo_h = int(h * 0.75)

        # Create intro video with logo fading in and out
        print("Creating logo intro video...")
        result1 = subprocess.run([
            "ffmpeg", "-y",
            "-loop", "1", "-i", self.logo_path,
            "-vf",
            f"scale={logo_w}:{logo_h}:flags=neighbor:force_original_aspect_ratio=decrease:eval=frame,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fade=t=in:st=0:d={self.fade_sec},"
            f"fade=t=out:st={self.intro_sec - self.fade_sec}:d={self.fade_sec},"
            "format=yuv420p",
            "-t", str(self.intro_sec),
            "-r", str(self.fps),
            "-c:v", "libx264",
            "-crf", "0",
            "-preset", "ultrafast",
            intro_mp4
        ], capture_output=True, text=True)

        if result1.returncode != 0:
            print(f"Error creating intro video: {result1.stderr}")
            return
        else:
            print("Logo intro video created successfully")

        # Create concatenation file
        with open(concat_txt, "w") as f:
            f.write(f"file '{os.path.abspath(intro_mp4)}'\n")
            f.write(f"file '{os.path.abspath(main_video)}'\n")

        # Concatenate videos
        print("Concatenating videos...")
        result2 = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_txt,
            "-c", "copy",
            final_video
        ], capture_output=True, text=True)

        if result2.returncode != 0:
            print(f"Error concatenating videos: {result2.stderr}")
            return
        else:
            print("Videos concatenated successfully")

        # Replace original video with the concatenated one
        try:
            if os.path.exists(final_video):
                os.replace(final_video, main_video)
                print(f"Final video with logo intro saved: {main_video}")
            else:
                print("Final video file not found")
        except Exception as e:
            print(f"Error replacing video file: {e}")

        # Clean up temporary files
        for temp_file in [intro_mp4, concat_txt]:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up {temp_file}: {e}")