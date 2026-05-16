from execute_LLM_plan import GoToObject, ExploreObject, Explore, PickupObject, PutObject
# from task_decorator import replace_explore_with_custom

# @replace_explore_with_custom

def pick_up_apple_and_put_it_on_coffee_table_in_living_room_on_floor_1(robot):
    # 0: go to the apple in the kitchen on floor 1.
    GoToObject(robot, {'object': 'Apple', 'room': 'kitchen', 'floor': 1})
    # 1: pick up the apple.
    PickupObject(robot, 'Apple')
    # 2: go to the coffee table in the living room on floor 1.
    GoToObject(robot, {'object': 'Coffee Table', 'room': 'living room', 'floor': 1})
    # 3: put the apple on the coffee table.
    PutObject(robot, 'Apple', 'Coffee Table')