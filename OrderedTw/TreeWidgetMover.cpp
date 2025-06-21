#include "TreeWidgetMover.h"
#include <QComboBox>
#include <QLineEdit>

TreeWidgetMover::TreeWidgetMover(QTreeWidget* tree, QPushButton* upBtn, QPushButton* downBtn, QObject* parent)
    : QObject(parent), m_tree(tree), m_upBtn(upBtn), m_downBtn(downBtn)
{
    connect(m_upBtn, SIGNAL(clicked()), this, SLOT(moveUp()));
    connect(m_downBtn, SIGNAL(clicked()), this, SLOT(moveDown()));
}

void TreeWidgetMover::moveUp()
{
    QTreeWidgetItem* current = m_tree->currentItem();
    if (!current) return;
    int currentIndex = m_tree->indexOfTopLevelItem(current);
    if (currentIndex > 0) {
        moveItemWithFreshWidgets(currentIndex, currentIndex - 1);
        m_tree->setCurrentItem(m_tree->topLevelItem(currentIndex - 1));
    }
}

void TreeWidgetMover::moveDown()
{
    QTreeWidgetItem* current = m_tree->currentItem();
    if (!current) return;
    int currentIndex = m_tree->indexOfTopLevelItem(current);
    if (currentIndex < m_tree->topLevelItemCount() - 1) {
        moveItemWithFreshWidgets(currentIndex, currentIndex + 1);
        m_tree->setCurrentItem(m_tree->topLevelItem(currentIndex + 1));
    }
}

void TreeWidgetMover::moveItemWithFreshWidgets(int from, int to)
{
    if (from < 0 || to < 0 || from >= m_tree->topLevelItemCount() || to > m_tree->topLevelItemCount())
        return;
    if (from == to) return;

    QTreeWidgetItem* item = m_tree->topLevelItem(from);

    // --- Store widget data ---
    // For this example, assuming col 1 is QComboBox, col 2 is QLineEdit
    int comboIndex = -1;
    QString lineEditText;

    QComboBox* combo = qobject_cast<QComboBox*>(m_tree->itemWidget(item, 1));
    if (combo) {
        comboIndex = combo->currentIndex();
    }
    QLineEdit* lineEdit = qobject_cast<QLineEdit*>(m_tree->itemWidget(item, 2));
    if (lineEdit) {
        lineEditText = lineEdit->text();
    }

    // Remove and delete old widgets (optional: deletion, since QTreeWidget is parent, but for safety)
    if (combo) {
        m_tree->removeItemWidget(item, 1);
        combo->deleteLater();
    }
    if (lineEdit) {
        m_tree->removeItemWidget(item, 2);
        lineEdit->deleteLater();
    }

    // Take and insert item
    item = m_tree->takeTopLevelItem(from);
    m_tree->insertTopLevelItem(to, item);

    // --- Re-create widgets and restore data ---
    QComboBox* newCombo = new QComboBox(m_tree);
    newCombo->addItems(QStringList() << "Option 1" << "Option 2" << "Option 3");
    if (comboIndex >= 0 && comboIndex < newCombo->count())
        newCombo->setCurrentIndex(comboIndex);
    m_tree->setItemWidget(item, 1, newCombo);

    QLineEdit* newLineEdit = new QLineEdit(m_tree);
    newLineEdit->setText(lineEditText);
    m_tree->setItemWidget(item, 2, newLineEdit);
}