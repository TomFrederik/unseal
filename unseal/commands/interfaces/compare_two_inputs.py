import os

from setuptools.cmd import Command

def main():
    os.system(f"streamlit run {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/interface/plain_interfaces/compare_two_inputs.py")