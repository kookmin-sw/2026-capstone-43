def GoToObject(robots, dest_obj):
    # Navigate to a fully specified target object.
    
    # Use this action only when the target object location is fully specified.
    # The target must include:
    # - object
    # - floor
    # - room or room_instance
    
    # If the instruction does not specify enough information,
    # use Explore(...) instead of GoToObject(...).
    
    # Example:
    # <Instruction> Go to the apple in the kitchen on floor 1.
    # Python script:
    # GoToObject(robot, {'object': 'Apple', 'room': 'kitchen', 'floor': 1})
    pass

def Explore(robot, sw_obj, position):
    # Explore is used when the target is underspecified.
    
    # Use Explore when the instruction does not provide enough information
    # to directly call GoToObject(...).
    
    # Example:
    # <Instruction> Find an apple.
    # Python script:
    # Explore(robot, {'object': 'Apple'})
    pass

def PickupObject(robot, pick_obj):
    # Pick up the object that was targeted by the most recent navigation step.
    
    # This action should be called after GoToObject(...) or Explore(...)
    # has already moved the robot to the intended object.
    
    # PickupObject does not independently search the whole scene.
    # Instead, it uses the target context established by the previous step
    # and resolves the corresponding simulator runtime object.
    
    # Example:
    # <Instruction> Pick up the apple in the kitchen on floor 1.
    # Python script:
    # GoToObject(robot, {'object': 'Apple', 'room': 'kitchen', 'floor': 1})
    # PickupObject(robot, 'Apple')       
    pass

def PutObject(robot, put_obj, recp): 
    # Put the currently held object on the receptacle targeted by the most recent navigation step.
    
    # This action assumes:
    # 1) the object has already been picked up
    # 2) the robot has already navigated to the destination receptacle
    
    # PutObject does not independently search the whole scene.
    # It uses the receptacle target context established by the previous
    # GoToObject(...) or Explore(...) step.
    
    # Example:
    # <Instruction> Put the apple on the table in the dining room on floor 1.
    # Python script:
    # GoToObject(robot, {'object': 'Table', 'room': 'dining room', 'floor': 1})
    # PutObject(robot, 'Apple', 'Table')
    pass