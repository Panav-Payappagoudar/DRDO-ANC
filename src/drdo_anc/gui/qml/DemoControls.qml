import QtQuick
import QtQuick.Layouts

Item {
    id: root
    property color white: "#F0F4F8"
    property color cyan: "#00E5FF"
    property color dim: "#666666"
    property color buttonBg: "#151A22"
    property color buttonActive: "#00E5FF"

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Text { text: "MODE"; color: dim; font.pixelSize: 10; font.bold: true }

            DemoButton {
                label: "DEMO"
                active: guiBridge.operationMode === "demo"
                onActivated: guiBridge.setDemoMode()
            }
            DemoButton {
                label: "LIVE"
                active: guiBridge.operationMode === "live"
                onActivated: guiBridge.setLiveMode()
            }

            Item { Layout.fillWidth: true }
            Text { text: guiBridge.demoScenario; color: cyan; font.pixelSize: 10 }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8
            visible: guiBridge.operationMode === "demo"

            Text { text: "SCENARIO"; color: dim; font.pixelSize: 10; font.bold: true }

            Repeater {
                model: guiBridge.scenarioLabels
                delegate: DemoButton {
                    label: (index + 1) + " " + modelData
                    active: guiBridge.selectedScenarioIndex === index
                    onActivated: guiBridge.selectScenario(index)
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Text { text: "TRANSPORT"; color: dim; font.pixelSize: 10; font.bold: true }
            DemoButton { label: "Play"; onActivated: guiBridge.play() }
            DemoButton { label: "Pause"; onActivated: guiBridge.pause() }
            DemoButton { label: "Stop"; onActivated: guiBridge.stop() }

            Item { Layout.fillWidth: true }

            Text {
                visible: guiBridge.operationMode === "demo"
                text: "A/B"
                color: dim
                font.pixelSize: 10
                font.bold: true
            }
            DemoButton {
                visible: guiBridge.operationMode === "demo"
                label: "A Raw"
                active: guiBridge.abMode === "raw"
                onActivated: guiBridge.selectAbRaw()
            }
            DemoButton {
                visible: guiBridge.operationMode === "demo"
                label: "B Enhanced"
                active: guiBridge.abMode === "enhanced"
                onActivated: guiBridge.selectAbEnhanced()
            }
        }

        PipelineChain {
            Layout.fillWidth: true
            Layout.preferredHeight: 24
        }
    }
}
