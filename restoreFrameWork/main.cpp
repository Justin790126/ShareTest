#include <QtGui> // Includes all necessary Qt4 classes
#include <QUndoStack>
#include <QUndoCommand>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QWidget>
#include <QShortcut>
#include <QPushButton>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

// Custom shape - a simple rectangle
class CustomShape : public QGraphicsRectItem {
public:
    CustomShape(qreal x, qreal y, qreal w, qreal h, QGraphicsItem *parent = nullptr)
        : QGraphicsRectItem(x, y, w, h, parent) {
        setBrush(QBrush(Qt::blue));  // Fill the rectangle with blue color
        setFlags(QGraphicsItem::ItemIsMovable);  // Allow the shape to be movable
    }
};

// Command to add a shape to the scene
class AddShapeCommand : public QUndoCommand {
public:
    AddShapeCommand(QGraphicsScene *scene, QPointF position, QUndoCommand *parent = nullptr)
        : QUndoCommand(QObject::tr("Add Shape"), parent), scene(scene), position(position), shape(nullptr) {}

    void undo() override {
        scene->removeItem(shape);  // Remove the shape from the scene
        delete shape;  // Deallocate the memory for the shape
        shape = nullptr;
    }

    void redo() override {
        if (!shape) {
            shape = new CustomShape(position.x(), position.y(), 50, 50);  // Create a new shape
        }
        scene->addItem(shape);  // Add the shape to the scene
    }

private:
    QGraphicsScene *scene;  // The graphics scene to add the shape to
    QPointF position;  // The position where the shape is added
    QGraphicsItem *shape;  // Pointer to the shape
};

// Main window to contain the scene, view, undo stack, and button
class MainWindow : public QWidget {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr) : QWidget(parent) {
        scene = new QGraphicsScene(this);  // The scene for shapes
        view = new QGraphicsView(scene, this);  // The view to display the scene
        undoStack = new QUndoStack(this);  // Undo stack to manage undo/redo actions
        addButton = new QPushButton(QObject::tr("Add Random Shape"), this);  // Button to add random shapes

        auto *layout = new QVBoxLayout(this);
        layout->addWidget(view);
        layout->addWidget(addButton);  // Add the button to the interface

        // Connect the button click to the addRandomShape() slot
        connect(addButton, SIGNAL(clicked()), this, SLOT(addRandomShape()));

        // Create a shortcut for undo (Ctrl+Z)
        undoShortcut = new QShortcut(QKeySequence("Ctrl+Z"), this);
        connect(undoShortcut, SIGNAL(activated()), this, SLOT(onUndoTriggered()));

        resize(800, 600);
        setWindowTitle(QObject::tr("Qt Undo Example"));

        scene->setSceneRect(0, 0, 800, 600);  // Set the scene dimensions

        std::srand(std::time(0));  // Seed the random number generator (for rand())
    }

public slots:
    // Undo the last command
    void onUndoTriggered() {
        undoStack->undo();
    }

    // Add a random shape to the scene
    void addRandomShape() {
        // Generate a random position within the scene dimensions
        qreal x = std::rand() % 750;  // Random x coordinate between 0 and 750
        qreal y = std::rand() % 550;  // Random y coordinate between 0 and 550
        undoStack->push(new AddShapeCommand(scene, QPointF(x, y)));  // Push a new AddShapeCommand to the undo stack
    }

private:
    QGraphicsView *view;         // Graphics view for displaying the scene
    QGraphicsScene *scene;       // Graphics scene containing shapes
    QUndoStack *undoStack;       // Stack to manage undo/redo actions
    QShortcut *undoShortcut;     // Shortcut for the undo action
    QPushButton *addButton;      // Button for adding random shapes
};

// Main function
int main(int argc, char **argv) {
    QApplication app(argc, argv);

    MainWindow window;
    window.show();

    return app.exec();
}

#include "main.moc"  // Include the Meta-Object Compiler file