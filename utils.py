def read_file(path):
  with open(path,'r') as tmp:
    filepaths = tmp.read().splitlines()
    return filepaths

def sec_to_hm(secs):
  hours = secs//3600
  minits = secs%3600//60
  secs = (secs - hours*3600 - minits*60)
  print_str = "{}h {}m {}s"
  return print_str.format(hours, minits, secs)