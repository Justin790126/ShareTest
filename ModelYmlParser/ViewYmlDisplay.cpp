#include "ViewYmlDisplay.h"


ViewYmlDisplay::ViewYmlDisplay(QWidget *parent)
    : QWidget(parent)
{

    Widgets();
    Layouts();
}

std::string test = R"(

# 123

* abc
* 2345
* juytbjhrvgrdr

----

**qqqqqqq**

*titl*


## 456
### 789

1. qaz
2. wsx
3. rfv

- dfghj
- ikj



````
#include <iostream>


using namespace std;

int main()
{
    cout << "hello " << endl;
    return 0;
}

````


Column 1 | Column 2
---------|---------
foo      | bar
baz      | qux
quux     | quuz


[title](https://www.google.com)

![alt text](./icon48.png)



![alt text](./icon48.png "Logo Title Text 1")



| Syntax | Description |
| ----------- | ----------- |
| Header | Title |
| Paragraph | Text |


- [ ] 2345
- [x] 9999

)";

void ViewYmlDisplay::Widgets()
{
    twYmlDisplay = new QTreeWidget();
    twYmlDisplay->setColumnCount(3); // Set the number of columns to 3
    QStringList headers;
    headers << "Key" << "Type" << "Value";
    twYmlDisplay->setHeaderLabels(headers);


    QTreeWidgetItem *topLevelItem = new QTreeWidgetItem(twYmlDisplay);
    topLevelItem->setText(0, "");
    twYmlDisplay->addTopLevelItem(topLevelItem);
    QPushButton *button = new QPushButton("Click me");

    // Set the button as the widget for the first column of the item
    twYmlDisplay->setItemWidget(topLevelItem, 0, button);


    teManual = new QTextEdit;

    
    int argc=3;
    char* argv[3] = {"md2html", "--github", "--html-css=style.css"};
    if(initMdParser(argc,argv) != 0) {
        exit(1);
    }
    char* html = process_string(test.c_str());
    printf("%s\n",html);
    teManual->setHtml(html);
}

void ViewYmlDisplay::Layouts()
{
    QVBoxLayout* vlytMain = new QVBoxLayout;
    vlytMain->setContentsMargins(0,0,0,0);
    {
        spltMain = new QSplitter(Qt::Vertical);
        spltMain->addWidget(twYmlDisplay);
        spltMain->addWidget(teManual);
    }
    vlytMain->addWidget(spltMain);
    setLayout(vlytMain);
}
