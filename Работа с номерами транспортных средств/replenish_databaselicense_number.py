with open('numbers.txt', 'a') as f:
	s = input('Введите номер в селедующем формате: A123BC45 --> ') + '\n'
	f.write(s)
	print("Номер успешно зарегистриован")