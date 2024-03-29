# c/c++ 基础

> 由于之前学过一部分了，这里只做关键点和盲点的记录

## 扫盲

### 宏定义

```cpp
#define F(x) ((x) +(x)))
#define PI 3.14159
```
宏定义就是在程序中用简单的文本替换代替一些复杂或重复的代码的方法。最好的一点就是它效率很高，因为本质不是一个函数，只是一个文本替代，没有传统函数调用的很多开销（保存寄存器，传递参数，下一条指令，函数返回，恢复寄存器值等）。
>[!note] 宏定义不是C++语言的一部分，而是由预处理器在编译时处理的。<br> 原理分析：<br> 预处理器在编译源代码之前执行，它会根据宏定义进行简单的文本替换。当编译器遇到源代码中的宏名称时，会将宏的定义替换为相应的文本。这个过程称为宏展开。宏展开是在编译之前完成的，因此宏的替换是纯文本替换，没有类型检查或其他编译器检查。

使用宏定义的常见问题：
1. 括号问题： 当定义带参数的宏时，务必在参数使用时使用括号，以确保在宏展开时不会导致意外的运算顺序问题。

2. 多次展开： 在宏定义中，如果有其他宏调用，宏可能会被多次展开，导致意外结果。这种情况下，可以使用##运算符进行粘合操作。

    ```cpp
    #define CONCAT(a, b) a ## b
    ```

>[!tip] 按照教程的说法，如果定义的宏是一个较为复杂的函数，最好使用do while来实现
>```cpp
>#define f() do{...;}while(0)
>```
>这样在后续调用宏函数f时就可以像正常的函数一样调用和使用分号且不存在冲突状况。 

### 常量、常量表达式

常量可以由两种方法表示：

1. 宏定义

2. const

> [!note]两者的区别在于宏定义是文本替换，而const是在程序内存中开辟一个不变的空间。

常量表达式：是指在编译阶段就可以直接求值的表达式。（宏定义就是常量表达式）

> 常量表达式通常可以用来获取数据的长度和switch的标签

宏定义和typedef的区别：

> 宏定义实在预处理阶段进行的，编译器无法识别宏定义；<br> typedef是能够被编译器识别，typedef通常用于定义别名，如果它发生错误可以被编译器识别，返回一些友好的提示。

### 字符操作

常用的字符函数：定义在\<ctype.h\>头文件内
```cpp
if (isalpha('A')) {
    // 字符 'A' 是一个字母
}
if (isdigit('5')) {
    // 字符 '5' 是一个数字
}
if (isalnum('7')) {
    // 字符 '7' 是一个字母或数字
}
if (islower('b')) {
    // 字符 'b' 是一个小写字母
}
if (isupper('X')) {
    // 字符 'X' 是一个大写字母
}
char lowercase = tolower('C'); // 将 'C' 转换为 'c'
char uppercase = toupper('d'); // 将 'd' 转换为 'D'
if (isspace(' ')) {
    // 字符 ' ' 是一个空白字符
}
```
字符读写问题：

1. 如何忽略输入输出时字符前面的空白字符的问题

    > 1：使用getchar()和putchar()，同时可以注意这两个效率比scanf和printf效率高<br> 
    > 2：惯用法：
    >```cpp
    >while(getchar() != '\n'); // 读取剩余的所有空格
    >```
    > 3：gets( )

### C语言的字符串

>[!note] C语言是没有专门的字符串类型的，是依赖于字符数组存在的。