#!/usr/bin/env python3
import argparse

def main(mode):
  import pdb; pdb.set_trace()
  x=1

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-m", "--mode", default='test')
  args = parser.parse_args()
  main(args.mode)
  
  
  
  
  
  
  
  
  
