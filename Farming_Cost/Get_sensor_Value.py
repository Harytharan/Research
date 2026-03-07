import serial
import time
import threading
import traceback
import sys

class SerialMonitor:
    def __init__(self, port="COM3", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False

    def connect(self):
        print("--- SENSOR MONITOR ---")
        print(f"Port    : {self.port}")
        print(f"Baudrate: {self.baudrate}\n")

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            print(f"Connected to {self.port}\n")
        except Exception:
            print("ERROR: Cannot open serial port!")
            traceback.print_exc()
            sys.exit(1)

    def start(self):
        print("Serial Listener...\n")
        thread = threading.Thread(target=self.read_loop, daemon=True)
        thread.start()

        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nStopping Serial Monitor...")
            self.running = False

    def read_loop(self):
        while self.running:
            try:
                raw = self.ser.readline()
                if raw:
                    # Decode serial
                    try:
                        text = raw.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        text = raw.decode(errors="ignore").strip()

                    # Print  output
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    if ":" in text:
                        key, value = text.split(":", 1)
                        print(f"[{timestamp}] {key.strip():<15} : {value.strip()}")
                    else:
                        print(f"[{timestamp}] {text}")

            except Exception:
                print("Serial Read Error:")
                traceback.print_exc()
                time.sleep(0.5)

if __name__ == "__main__":
    monitor = SerialMonitor()
    monitor.connect()
    monitor.start()
