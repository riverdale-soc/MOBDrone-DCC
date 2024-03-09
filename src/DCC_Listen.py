"""
Author: Sophia Ventresca
ESP32 Submersion Handler UART Listener
Opens USB-Serial Port and waits for Wake String, then constructs MOB Wake Dict
"""

#import serial
import time
import enum

# Define exceptions for MOB Parser, Invalid state, invalid format, etc.
class MOBParserException(Exception):
    pass

# Enum of valid MOB States
class MOB_STATE(enum.Enum):
    MOB_WAKE = "WAKE"
    MOB_RESET = "RESET"
    MOB_NONE = "NONE"

class DCCListener:
    def __init__(self):
        self.port = None

        self.mob_wake_dict = {
            'MOB_STATE' : MOB_STATE.MOB_NONE, 
            'Altitude' : "",
            'Longitude' : "",
            'Latitude' : "",
        }

        self.mob_time = None # Time stamp


    def init_port(self) -> bool:
        """
          Open Serial Port to ESP32. 
          UART Configs: 115200 Baud, 8N1, XON/XOFF, PARITY NONE, RTS/CTS, 
        """
        self.port = serial.Serial('/dev/ttyUSB0', baudrate=115200, xonxoff=True)
        if self.port.is_open is not True:
            self.port.open()
        return True

    # Read all lines from serial port, and parse for MOB Wake String
    def listen(self) -> None:
        if self.port is None:
            raise Exception("Serial Port is not initialized, init port first")
        while True:
            line = self.port.readline().decode('utf-8')
            if self.parse_line(line) is True:
                # Implement wake up logic here
                print("MOB Wake String Received") 
                # Return tuple of MOB Wake Dict arguments and Time Stamp as (MOB_STATE, Altitude, Longitude, Latitude, Time Stamp)
                return (self.mob_wake_dict['MOB_STATE'], self.mob_wake_dict['Altitude'], self.mob_wake_dict['Longitude'], self.mob_wake_dict['Latitude'], self.mob_time)


    # Parse line for MOB Wake String
    # We will assume it will come in a single line like: 
    # MOB indicates MOB Wake String, MOB State, Altitude, Longitude, Latitude
    #        "MOB, OK, 12345.12345, 12345.12345, 12345.12345"
    def parse_line(self, line: str) -> bool:
        """
         Read line from serial port
         Parse line for MOB Wake String, if true, update MOB Wake Dict
         if it is not a MOB Wake String, return false
         if the state is invalid, raise exception
         Split line into list of strings, serperated by commas
         remove any whitespaces from strings
        """
        mob_pattern = [x.strip() for x in line.split(',')]
        # Check if first string is MOB
        if mob_pattern[0] != "MOB":
            return False
        if len(mob_pattern) != 5:
            raise MOBParserException("Invalid MOB Wake String Format")
        
        # Check if second string is valid MOB State
        if mob_pattern[1] == "WAKE":
            self.mob_wake_dict['MOB_STATE'] = MOB_STATE.MOB_WAKE
        elif mob_pattern[1] == "RESET":
            self.mob_wake_dict['MOB_STATE'] = MOB_STATE.MOB_RESET
        else:
            raise MOBParserException("Invalid MOB State")
        
        # Check if Altitude is valid float
        try:
            float(mob_pattern[2])
        except ValueError:
            raise MOBParserException("Invalid Altitude")
        self.mob_wake_dict['Altitude'] = mob_pattern[2]

        # Check if Longitude is valid float
        try:
            float(mob_pattern[3])
        except ValueError:
            raise MOBParserException("Invalid Longitude")
        self.mob_wake_dict['Longitude'] = mob_pattern[3]

        # Check if Latitude is valid float
        try:
            float(mob_pattern[4])
        except ValueError:
            raise MOBParserException("Invalid Latitude")
        self.mob_wake_dict['Latitude'] = mob_pattern[4]

        self.mob_time = time.time()
        return True
 
    # print MOB Wake Dict by iterating through keys and values
    def print_mob_wake_dict(self):
        print("MOB String Received at: ", self.mob_time)
        for key, value in self.mob_wake_dict.items():
            print(key, value)

    def write_serial(self, data: str) -> bool:
        if self.port is None:
            raise Exception("Serial Port is not initialized, init port first")
        # Convert to bytes
        data = data.encode('utf-8')
        self.port.write(data)
        return 0

    def close_port(self) -> bool:
        if self.port is None:
            raise Exception("Serial Port is not initialized, init port first")
        self.port.close()
        return 0

# Main function to test DCCListener
# Open Serial Port, listen for MOB Wake String, print MOB Wake Dict
if __name__ == '__main__':
    dcclisten = DCCListener()
    dcclisten.init_port()
    dcclisten.listen()