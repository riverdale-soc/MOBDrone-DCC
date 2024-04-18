import asyncio
from mavsdk import System
from DCC_Listen import DCC_Listen
from DCC_MissionBuilder import Mission, LocationGlobal


"""
- Main DCC Application for simulation of response to MOB event in Gazebo environment
Represents states:
    
"""

targetAlt = 30

    
async def run():
    drone = System()
    dcclisten = DCC_Listen()
    dcclisten.init_port()
    print("Listening for MOB Event")
    state, _, longitude, latitude, timestamp = dcclisten.listen()
    # Start a connection to drone
    await drone.connect(system_address="udp://:14540")
    # Indicates ESP-NOW packet does not include GPS coordinates
    if longitude == 0 and latitude == 0:
        gps_arrived = False
    else:
        gps_arrived = True
    print("MOB String Received at: ", timestamp)

    home = drone.telemetry.home()

    # Create Mission Object where LocationGlobal is a home coordinates with targetAlt as altitude
    mission = Mission(LocationGlobal(home.long, home.lat, targetAlt), 30, 100, 160, 150*150)
    mission.build_mission(longitude, latitude, targetAlt,
                      Area=10*10,        # Area to search in meters 
                      Cam_FOV=160,       # Camera Field of View
                      MAX_RANGE=150*150)


    # Check if vehicle is armable
    async for is_armable in drone.telemetry.armed():
        print(f"Is armable: {is_armable}")
        break

    # Check if vehicle is armed
    async for is_armed in drone.telemetry.armed():
        print(f"Is armed: {is_armed}")
        break

    async def callback(is_armable):
        print(f"Is armable: {is_armable}")
        print(" Battery: ", await drone.telemetry.battery())
        print(" GPS: ", await drone.telemetry.gps_info())
        print(" System status: ", await drone.telemetry.status_text())
        print(" Attitude: ", await drone.telemetry.attitude_euler())
        print(" Position: ", await drone.telemetry.position())
        print(" Ground speed: ", await drone.telemetry.ground_speed())
        print(" Heading: ", await drone.telemetry.heading())
        print(" Home position: ", await drone.telemetry.home())
        print(" In air: ", await drone.telemetry.in_air())
        print(" Landed state: ", await drone.telemetry.landed_state())
        print(" RC status: ", await drone.telemetry.rc_status())
        print(" Timestamp: ", await drone.telemetry.timestamp())
        print(" Health: ", await drone.telemetry.health())
        print(" Connection info: ", await drone.telemetry.health())
        print(" Arming status: ", await drone.telemetry.armed())
        print(" Flight mode: ", await drone.telemetry.flight_mode())
        print(" Last Heartbeat: ", await drone.telemetry.last_heartbeat())

    drone.subscribe_telemetry(callback)

    async def wait():
        while not await drone.telemetry.health_all_ok():
            print("Waiting for system to be ready")
            await asyncio.sleep(1)


    async def arm_and_takeoff(target_altitude):
        print("Basic pre-arm checks")
        # Don't let the user try to arm until autopilot is ready

        async def wait_for_armable():
            while not await drone.telemetry.health_all_ok():
                print("Waiting for system to be ready")
                await asyncio.sleep(1)

        await wait_for_armable()

        print("Arming motors")
        # Copter should arm in GUIDED mode
        await drone.action.set_takeoff_altitude(target_altitude)
        await drone.action.arm()
        await drone.action.takeoff()

        # Confirm vehicle armed before attempting to take off
        while not await drone.telemetry.armed():
            print(" Waiting for arming...")
            await asyncio.sleep(1)

        print("Taking off!")

        # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
        async def wait_until_altitude_reached():
            while (await drone.telemetry.altitude()).relative_altitude_m < target_altitude * 0.95:
                print(" Altitude: ", (await drone.telemetry.altitude()).relative_altitude_m)
                await asyncio.sleep(1)

        await wait_until_altitude_reached()

    await arm_and_takeoff(targetAlt)

    print("Set default/target airspeed to 3")

    await drone.action.set_maximum_speed(3)
    await drone.action.set_return_to_launch_altitude(targetAlt)

    if gps_arrived:
        point = [longitude, latitude, targetAlt]
        print("Going to first point for MOB")
        await drone.action.goto_location(*point)

    await asyncio.sleep(10)

    await asyncio.gather(wait(), asyncio.sleep(5))



    # Close the drone
    await drone.close()

if __name__ == "__main__":
    asyncio.run(run())
