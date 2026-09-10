import QtQuick
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls

Window {
    id: mainWindow
    width: 1024
    height: 900
    visible: true
    title: qsTr("DRDO-ANC Telemetry Console")

    Shortcut { sequence: "Space"; onActivated: { if (guiBridge.playbackState === "playing") guiBridge.pause(); else guiBridge.play() } }
    Shortcut { sequence: "A"; onActivated: guiBridge.selectAbRaw() }
    Shortcut { sequence: "B"; onActivated: guiBridge.selectAbEnhanced() }
    Shortcut { sequence: "1"; onActivated: guiBridge.selectScenario(0) }

    // Deep Premium Dark Palette
    property color black: "#05070A"
    property color white: "#F0F4F8"
    property color cyan: "#00E5FF"
    property color darkGrey: "#0F1218"
    property color lightGrey: "#2A2E35"
    
    Rectangle {
        anchors.fill: parent
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#0B101E" }
            GradientStop { position: 1.0; color: "#000000" }
        }
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 16
        spacing: 16
        
        // Header
        RowLayout {
            Layout.fillWidth: true
            
            Text {
                text: "DRDO-ANC"
                color: white
                font.pixelSize: 48
                font.bold: true
                font.letterSpacing: -1
                Layout.alignment: Qt.AlignLeft
            }
            
            Text {
                text: guiBridge.isLive ? "ACTIVE" : (guiBridge.operationMode === "demo" ? "DEMO" : "OFFLINE")
                color: cyan
                font.pixelSize: 48
                font.bold: true
                Layout.alignment: Qt.AlignLeft
            }
            
            Item { Layout.fillWidth: true }
            
            // System properties small text
            ColumnLayout {
                spacing: 2
                Layout.alignment: Qt.AlignRight
                Text { text: "MODEL: " + guiBridge.modelName; color: lightGrey; font.pixelSize: 10 }
                Text { text: "SAMPLE RATE: " + guiBridge.sampleRate; color: lightGrey; font.pixelSize: 10 }
                Text { text: "GUI FPS: 60"; color: lightGrey; font.pixelSize: 10 }
            }
        }
        
        Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: lightGrey }

        Text {
            Layout.fillWidth: true
            visible: guiBridge.errorMessage.length > 0
            text: "ERROR: " + guiBridge.errorMessage
            color: "#FF5577"
            font.pixelSize: 12
            wrapMode: Text.WordWrap
        }

        DemoControls {
            Layout.fillWidth: true
            Layout.preferredHeight: 120
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 16

            DemoPanel {
                Layout.preferredWidth: 260
                Layout.fillHeight: true
            }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 16

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            text: guiBridge.operationMode === "demo"
                                ? guiBridge.demoInputLabel
                                : "RAW MIC INPUT"
                            color: white
                            font.pixelSize: 18
                            font.bold: true
                            font.letterSpacing: 1
                        }
                        Item { Layout.fillWidth: true }
                        Rectangle {
                            color: "transparent"
                            border.color: cyan
                            border.width: 1
                            Layout.preferredWidth: statusText.width + 8
                            Layout.preferredHeight: statusText.height + 4
                            Text { id: statusText; text: "STATUS: CAPTURING"; color: cyan; font.pixelSize: 10; anchors.centerIn: parent }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "#08FFFFFF"
                        border.color: "#15FFFFFF"
                        border.width: 1
                        radius: 8

                        Waveform {
                            id: inputWaveform
                            anchors.fill: parent
                            lineColor: cyan
                        }
                    }

                    Connections {
                        target: guiBridge
                        function onInputWaveformUpdated(data) { inputWaveform.updateData(data); }
                    }
                }

                Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: lightGrey }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    spacing: 4

                    RowLayout {
                        Layout.fillWidth: true
                        Text {
                            text: guiBridge.operationMode === "demo"
                                ? guiBridge.demoOutputLabel
                                : "CLEAN ENHANCED OUTPUT"
                            color: white
                            font.pixelSize: 18
                            font.bold: true
                            font.letterSpacing: 1
                        }
                        Item { Layout.fillWidth: true }
                        Text {
                            text: "ENHANCED"; color: black; font.pixelSize: 10; font.bold: true; padding: 4
                            Rectangle { anchors.fill: parent; color: cyan; radius: 4; z: -1 }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "#08FFFFFF"
                        border.color: "#15FFFFFF"
                        border.width: 1
                        radius: 8

                        Waveform {
                            id: outputWaveform
                            anchors.fill: parent
                            lineColor: cyan
                        }
                    }

                    Connections {
                        target: guiBridge
                        function onOutputWaveformUpdated(data) { outputWaveform.updateData(data); }
                    }
                }

                Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: lightGrey }

                Metrics {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 250
                }
            }
        }
    }
}
