import threading
from tkinter import filedialog
import main as m
import tkinter as tk


def process_file(file_path):
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Extracting frames from video, it might take a few minutes, please wait...\n")
    output_dir = m.frame_capture(file_path,
                                 sample_rate=15,
                                 blurr_threshold=450,
                                 similarity_threshold=0.28,
                                 min_group_size=10)
    if output_dir:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "Video frames extracted and filtered.\n")
    else:
        result_text.insert(tk.END, "Processing stopped.\n")


def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_file(file_path)


def stop_processing():
    m.is_processing = False
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Stopping....\n")


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Files Antivirus")
    root.configure(bg='black')
    root.resizable(False, False)

    file_button = tk.Button(root, text="Upload Video", command=lambda: threading.Thread(target=select_file).start(),
                            height=3, width=15)
    file_button.pack(pady=10)

    dir_button = tk.Button(root, text="Stop Processing", fg="red", command=stop_processing, height=1, width=15)
    dir_button.pack(pady=10)

    result_text = tk.Text(root, wrap='word', height=15, width=50)
    result_text.pack(pady=10)
    result_text.insert(tk.END, "Please upload your video file.\n")
    root.mainloop()
