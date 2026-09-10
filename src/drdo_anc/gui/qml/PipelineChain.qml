import QtQuick
import QtQuick.Layouts

Item {
    id: root
    property color white: "#F0F4F8"
    property color cyan: "#00E5FF"
    property color dim: "#555555"
    property color active: "#00E5FF"
    property color inactive: "#333333"

    readonly property var stages: ["input", "capture", "stream", "df3", "output"]
    readonly property var labels: ["INPUT", "CAPTURE", "STREAM", "DeepFilterNet3", "OUTPUT"]

    RowLayout {
        anchors.fill: parent
        spacing: 6

        Repeater {
            model: labels.length
            delegate: RowLayout {
                spacing: 6
                Layout.alignment: Qt.AlignVCenter

                Rectangle {
                    Layout.preferredWidth: 8
                    Layout.preferredHeight: 8
                    radius: 4
                    color: stages[index] === guiBridge.pipelineStage ? active : inactive
                }

                Text {
                    text: labels[index]
                    color: stages[index] === guiBridge.pipelineStage ? white : dim
                    font.pixelSize: 9
                    font.bold: stages[index] === guiBridge.pipelineStage
                }

                Text {
                    visible: index < labels.length - 1
                    text: "↓"
                    color: dim
                    font.pixelSize: 8
                }
            }
        }
    }
}
