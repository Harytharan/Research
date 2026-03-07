import serial
import time
import threading
import traceback
import sys

class SerialMonitor:
    def __init__(self):
        self.port = "COM3"
        self.baudrate = 115200

        self.ser = None
        self.running = False

    def connect(self):
        print("SERIAL MONITOR")

        print(f"Using HARD-CODED PORT : {self.port}")
        print(f"⚙Using HARD-CODED BAUD : {self.baudrate}")


        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            self.running = True
            print(f"Connected to {self.port} successfully\n")
        except Exception as e:
            print("ERROR: Cannot open serial port!")
            traceback.print_exc()
            sys.exit(1)

    def start(self):
        print("Starting Serial Listener Thread...\n")
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
                    try:
                        text = raw.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        text = raw.decode(errors="ignore").strip()

                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    print(f"Timestamp : {timestamp}")
                    print(f"Raw Bytes : {raw}")
                    print(f"Decoded   : {text}")

            except Exception:
                print("Serial Read Error:")
                traceback.print_exc()
                time.sleep(0.5)

if __name__ == "__main__":
    monitor = SerialMonitor()
    monitor.connect()
    monitor.start()
