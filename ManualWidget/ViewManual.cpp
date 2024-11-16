#include "ViewManual.h"
#include <QVector>

ViewManual::ViewManual(QWidget *parent)
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
foo      | <span style='color:blue'>bar</span>
baz      | qux
quux     | quuz


[title](https://www.google.com)

![alt text](./icon48.png)



![alt text](./icon48.png "Logo Title Text 1")



| <span style='color:blue'>Syntax</span> | <span style='color:blue'>Description</span> |
| ----------- | ----------- |
| <span style='color:red'>Header</span> | Title |
| Paragraph | Text |


- [ ] 2345
- [x] 9999

)";

void ViewManual::Widgets()
{
    stkwManualPages = new QStackedWidget;
    lwManualTitle = new QListWidget;
    // twYmlDisplay = new QTreeWidget();
    // twYmlDisplay->setColumnCount(3); // Set the number of columns to 3
    // QStringList headers;
    // headers << "Key" << "Type" << "Value";
    // twYmlDisplay->setHeaderLabels(headers);


    // QTreeWidgetItem *topLevelItem = new QTreeWidgetItem(twYmlDisplay);
    // topLevelItem->setText(0, "");
    // twYmlDisplay->addTopLevelItem(topLevelItem);
    // QPushButton *button = new QPushButton("Click me");

    // // Set the button as the widget for the first column of the item
    // twYmlDisplay->setItemWidget(topLevelItem, 0, button);


    teManual = new QTextEdit;

    
    int argc=3;
    char* argv[3] = {"md2html", "--github", "--html-css=style.css"};
    if(initMdParser(argc,argv) != 0) {
        exit(1);
    }
    char* html = process_string(test.c_str());
    // std::string teststr = "<span style='color:red'> # abcde </span>";

    // string test1 = "# <span style='color:red'>123</span>";
    // std::string result;
    // // split_html_tag(teststr, sttag, content, endtag);
    
    // // split_html_tag1(test1,result);
    // // printf("result = %s\n", result.c_str());
    // // char* mdhtml = process_string(content.c_str());
    // // char html[512];
    // // sprintf(html, "%s%s%s", sttag.c_str(), mdhtml, endtag.c_str());
    // char* html = process_string(test.c_str());
    // // printf("%s\n",html);s
    teManual->setHtml(html);
}

QWidget* ViewManual::addManualPage(const string& btnText)
{
    if (!vlytManualTitle) return NULL;
    if (!vlytManualContent) return NULL;
    if (!stkwManualPages) return NULL;
    QWidget* content = new QWidget;
    // new QLabel(btnText.c_str(), content);
    lwManualTitle->addItem(btnText.c_str());
    connect(lwManualTitle, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(handleListWidgetItemClick(QListWidgetItem *)));
    stkwManualPages->addWidget(content);

    return content;
}

void ViewManual::handleListWidgetItemClick(QListWidgetItem *item)
{
    int index = lwManualTitle->row(item);
    stkwManualPages->setCurrentIndex(index);
}

void ViewManual::Layouts()
{
    QVBoxLayout* vlytMain = new QVBoxLayout;
    vlytMain->setContentsMargins(0,0,0,0);
    {
        hlytToolbar = new QHBoxLayout;
        hlytToolbar->setContentsMargins(0,0,0,0);
        {
            cbbSearchBar = new QComboBox;
            btnSearch = new QPushButton("Search");
        }
        hlytToolbar->addWidget(cbbSearchBar,5);
        hlytToolbar->addWidget(btnSearch,2);

        hlytManualMain = new QHBoxLayout;
        hlytManualMain->setContentsMargins(0,0,0,0);
        {  
            
            vlytManualTitle = new QVBoxLayout;
            {
                vlytManualTitle->addWidget(lwManualTitle);
            }
            vlytManualContent = new QVBoxLayout;
            {
                vlytManualContent->addWidget(stkwManualPages);
            }
            QWidget* content1 = addManualPage("test1");
            QVBoxLayout* lytContent1=new QVBoxLayout(content1);
            lytContent1->addWidget(teManual);

            QWidget* content2 = addManualPage("test2");
            QVBoxLayout* lytContent2=new QVBoxLayout(content2);
            lytContent2->addWidget(new QLabel("page2"));

            hlytManualMain->addLayout(vlytManualTitle, 3);
            hlytManualMain->addLayout(vlytManualContent, 7);
        }
        vlytMain->addLayout(hlytToolbar);
        vlytMain->addLayout(hlytManualMain);
    }
    setLayout(vlytMain);
}
