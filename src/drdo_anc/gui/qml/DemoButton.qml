import QtQuick

Rectangle {
    id: root
    property string label: ""
    property bool active: false
    signal activated()

    width: labelItem.width + 16
    height: 28
    radius: 4
    color: active ? "#00E5FF" : "#151A22"
    border.color: active ? "#00E5FF" : "#2A2E35"

    Text {
        id: labelItem
        anchors.centerIn: parent
        text: root.label
        color: active ? "#000000" : "#F0F4F8"
        font.pixelSize: 10
        font.bold: active
    }

    MouseArea {
        anchors.fill: parent
        onClicked: root.activated()
    }
}
