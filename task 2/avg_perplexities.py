

total = 0.0
length = 0.0
average = 0.0
infile = open('perplexities.txt', 'r')
contents = infile.readlines()
for line in contents:
    amount = float(line)
    if amount > 1000000.0:
        print(amount)
    total += amount
    length = length + 1

average = total / length
infile.close()
print('There were ', length, ' numbers in the file.' )
print(format(average, ',.2f'))
