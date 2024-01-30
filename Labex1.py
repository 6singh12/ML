#1
def count_pairs_with_sum(lst, total_sum):
    count = 0
    seen_numbers = set()

    for num in lst:
        complement = total_sum - num
        if complement in seen_numbers:
            count += 1
        seen_numbers.add(num)

    return count

given_list = [2, 7, 4, 1, 3, 6]
req_sum = 10
result = count_pairs_with_sum(given_list, req_sum)
print(f"Number of pairs with sum equal to {req_sum}: {result}")


#2
def calculate_range(lst):
    if len(lst) < 3:
        print('Range determination not possible')
    
    max_val = max(lst)
    min_val = min(lst)
    range_val = max_val-min_val

    return range_val

given_lst = [5,3,8,1,0,4]
result= calculate_range(given_lst)
print(f'Range : {result}')


#3
def multiply_matrices(matrix1, matrix2):
    result = [[0] * len(matrix2[0]) for _ in range(len(matrix1))]
    
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    
    return result

def matrix_power(A, m):
    if not isinstance(A, list) or not all(isinstance(row, list) and len(row) == len(A) for row in A) or len(A) == 0 or len(A) != len(A[0]):
        raise ValueError("Input must be a square matrix.")
    
    if m == 0:
        return [[1 if i == j else 0 for j in range(len(A))] for i in range(len(A))]

    result_matrix = A
    for _ in range(m - 1):
        result_matrix = multiply_matrices(result_matrix, A)

    return result_matrix

matrix_A = [[2, 1], [1, 2]]
exponent_m = 3

try:
    result_matrix = matrix_power(matrix_A, exponent_m)
    print(f"A raised to the power of {exponent_m}:\n{result_matrix}")
except ValueError as e:
    print(e)



#4
def highest_occurrence(input_string):
    char_count = {}

    for char in input_string:
        if char.isalpha():
            char_count[char] = char_count.get(char, 0) + 1

    max_char = max(char_count, key=char_count.get)
    max_count = char_count[max_char]

    return max_char, max_count

input_string = "hippopotamus"
result_char, result_count = highest_occurrence(input_string)

print(f"The highest occurring character is '{result_char}' with occurrence count {result_count}.")
