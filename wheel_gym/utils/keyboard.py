import isaacgym
import torch
from wheel_gym.envs.wheel.wheel import WheelRobot
from wheel_gym.envs.wheel.wheel_config import WheelRobotCfg
from wheel_gym.utils.math import *
from isaacgym.gymapi import (
    KEY_F,
    KEY_P,
    KEY_L,
    KEY_J,
    KEY_R,
    KEY_U,
    KEY_W,
    KEY_S,
    KEY_A,
    KEY_D,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    KEY_DOWN,
    KEY_X,
    KEY_SPACE,
    KEY_M,
    KEY_N,
    KEY_H
)

class KeyboardCtrl:
    def __init__(self, env: WheelRobot, env_cfg: WheelRobotCfg, **kwargs):
        self.env = env
        self.env_cfg = env_cfg
        self.record = False

        key_actions = {
            KEY_W: "forward",
            KEY_S: "backward",
            KEY_A: "leftturn",
            KEY_D: "rightturn",

            KEY_UP: "move_up",
            KEY_DOWN: "move_down",

            KEY_P: "push_robot",
            KEY_L: "press_robot",
            KEY_J: "action_jitter",
            KEY_R: "agent_full_reset",
            KEY_U: "full_reset",


            KEY_X: "stop",
            KEY_SPACE: "jump",
            KEY_M: "jump_higher",
            KEY_N: "jump_lower",
            KEY_F: "FPV",
            KEY_H: "record"
        }

        for key, action in key_actions.items():
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)

        # 跳跃高度值
        # self.jump_height = torch.full((1,), env_cfg.commands.ranges.jump_height[1])

    def run(self):
        # command: lin_vel_x, height, ang_vel_yaw, heading
        for ui_event in self.env.gym.query_viewer_action_events(self.env.viewer):
            if ui_event.value == 0:
                continue
            if ui_event.action == "forward":
                self.env.commands[:, 0] = 1.0
                self.env.commands[:, 0] = torch.clip(self.env.commands[:, 0], self.env.command_ranges["lin_vel_x"][0], self.env.command_ranges["lin_vel_x"][1])
            if ui_event.action == "backward":
                self.env.commands[:, 0] = -1.0
                self.env.commands[:, 0] = torch.clip(self.env.commands[:, 0], self.env.command_ranges["lin_vel_x"][0], self.env.command_ranges["lin_vel_x"][1])
            if ui_event.action == "move_up":
                self.env.commands[:, 1] += 0.05
                self.env.commands[:, 1] = torch.clip(self.env.commands[:, 1], self.env.command_ranges["height"][0], self.env.command_ranges["height"][1])
            if ui_event.action == "move_down":
                self.env.commands[:, 1] -= 0.05
                self.env.commands[:, 1] = torch.clip(self.env.commands[:, 1], self.env.command_ranges["height"][0], self.env.command_ranges["height"][1])
            if ui_event.action == "leftturn":
                self.env.commands[:, 3] = angle_operate(self.env.commands[:, 3], 0.5)
                #self.env.commands[:, 3] = wrap_to_pi(self.env.commands[:, 3])
            if ui_event.action == "rightturn":
                self.env.commands[:, 3] = angle_operate(self.env.commands[:, 3], -0.5)
            if ui_event.action == "jump":
                self.env.commands[:, 4] = 0.3
            if ui_event.action == "push_robot":
                self.env._push_robots()
            if ui_event.action == "stop":
                self.env.commands[:, 0] = 0
                self.env.commands[:, 2] = 0
                self.env.commands[:, 3] = 0
                self.env.commands[:, 4] = 0
            # if ui_event.action == "jump":
            #     self.env.commands[:, 5] = 0
            #     self.env.commands[:, 4] = self.jump_height
            # if ui_event.action == "jump_higher":
            #     self.jump_height += 0.05
            # if ui_event.action == "jump_lower":
            #     self.jump_height -= 0.05
            if ui_event.action == "record":
                self.record = not self.record
            self.print_command()
            # print(
            #     "v:{:.1f} h:{:.2f} a:{:.2f} head:{:.1f}".format(
            #         self.env.commands[:, 0].item(),
            #         self.env.commands[:, 1].item(),
            #         self.env.commands[:, 2].item(),
            #         self.env.commands[:, 3].item(),
            #     )
            # )


        
    def print_command(self):
        print(
            "v:{:.1f} h:{:.2f} a:{:.2f} head:{:.1f}".format(
                self.env.commands[0, 0].item(),
                self.env.commands[0, 1].item(),
                self.env.commands[0, 2].item(),
                self.env.commands[0, 3].item(),
            )
        )