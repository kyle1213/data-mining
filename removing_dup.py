with open('html_path_list.txt', 'r') as f:
    lines = f.readlines()

print(len(lines))
lines = set(lines)
lines = list(lines)
print(len(lines))

print(str(lines[0]))
with open('html_path_list_post_process.txt', 'wb') as f:
    for line in lines:
        f.write(line.encode('utf-8'))
