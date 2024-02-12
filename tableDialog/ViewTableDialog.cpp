#include "ViewTableDialog.h"

ViewTableDialog::ViewTableDialog(QWidget *parent)
    : QDialog(parent)
{
    // Create the layout for the dialog
    QVBoxLayout *mainLayout = new QVBoxLayout(this);

    // Create a horizontal layout for buttons
    QHBoxLayout *buttonLayout = new QHBoxLayout;

    // Create buttons
    okButton = new QPushButton("OK");
    cancelButton = new QPushButton("Cancel");
    applyButton = new QPushButton("Apply");

    // Connect signals and slots
    connect(okButton, SIGNAL(clicked()), this, SLOT(accept()));
    connect(cancelButton, SIGNAL(clicked()), this, SLOT(reject()));
    connect(applyButton, SIGNAL(clicked()), this, SLOT(applyChanges()));

    // Add buttons to button layout
    buttonLayout->addStretch(7);
    buttonLayout->addWidget(cancelButton,1);
    buttonLayout->addWidget(okButton,1);
    buttonLayout->addWidget(applyButton,1);

    // Create a horizontal layout for the dialog content
    QHBoxLayout *contentLayout = new QHBoxLayout;

    // Create a QTreeView on the left side
    treeView = new QTreeView;
    // Set up the tree model and populate it if needed
    QStandardItemModel *model = new QStandardItemModel;
    treeView->setModel(model);
    // Add items to the model if needed



    // Add widgets to the content layout
    contentLayout->addWidget(treeView,3);
    QScrollArea *scrollArea = new ViewColEdit();
    contentLayout->addWidget(scrollArea,7);

    // Add the content layout and button layout to the main layout
    mainLayout->addLayout(contentLayout);
    mainLayout->addLayout(buttonLayout);
}

void ViewTableDialog::applyChanges()
{
    // Implement any actions to apply changes here
}