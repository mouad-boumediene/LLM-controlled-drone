#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PromptChat(Node):
    def __init__(self) -> None:
        super().__init__("prompt_chat")
        self.pub = self.create_publisher(String, "/user_command", 10)

    def wait_for_subscriber(self, timeout_sec: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_sec:
            if self.count_subscribers("/user_command") > 0:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
        return False

    def send_prompt(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.1)


def main() -> None:
    rclpy.init()
    node = PromptChat()

    print("Drone chat started.")
    print("Type a prompt and press Enter.")
    print("Examples:")
    print("  take off and hover at 20 metres")
    print("  fly a 20 metre square pattern")
    print("  return to home and land")
    print("Type 'exit' or 'quit' to close.\n")

    if node.wait_for_subscriber():
        print("Connected to /user_command subscriber.\n")
    else:
        print("Warning: no subscriber detected on /user_command yet.")
        print("You can still type prompts, but make sure drone_agent.launch.py is running.\n")

    try:
        while True:
            try:
                text = input("drone> ").strip()
            except EOFError:
                print()
                break

            if not text:
                continue

            if text.lower() in {"exit", "quit"}:
                break

            node.send_prompt(text)
            print(f"sent: {text}\n")

    except KeyboardInterrupt:
        print("\nStopping chat...")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
