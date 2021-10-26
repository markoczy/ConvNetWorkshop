package main

import (
	"encoding/base64"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"
)

var (
	script    string
	modelRoot string
)

func indexHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Add("Content-Type", "text/html")
	http.ServeFile(w, r, "static/index.html")
}

func uploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer file.Close()
	d, err := ioutil.ReadAll(file)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	name, err := saveFile(d)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	fmt.Println(name)
	labels := getLabels(name)
	fmt.Println(labels)
	t, _ := template.ParseFiles("static/view.html")
	encoded := base64.RawStdEncoding.EncodeToString(d)
	_ = t.Execute(w, struct {
		Name    string
		Encoded string
		Labels  []string
	}{
		Name:    name,
		Encoded: encoded,
		Labels:  labels,
	})
}

func saveFile(d []byte) (string, error) {
	tempFile, err := ioutil.TempFile("upload", "upload-*.png")
	if err != nil {
		fmt.Println(err)
	}
	defer tempFile.Close()
	tempFile.Write(d)
	return filepath.Abs(tempFile.Name())
}

func getLabels(file string) []string {
	fmt.Println("CMD:", "python", script, modelRoot, file)
	cmd := exec.Command("python", script, modelRoot, file)
	out, err := cmd.Output()
	if err != nil {
		panic(err)
	}
	ret := []string{}
	for _, s := range strings.Split(string(out), "\n") {
		cur := strings.TrimSpace(s)
		if cur != "" {
			ret = append(ret, cur)
		}
	}
	return ret
}

func main() {
	scriptPtr := flag.String("script", "../../python/labeler.py", "Script path")
	modelRootPtr := flag.String("model-root", "../../model", "Root dir of models")
	flag.Parse()
	modelRoot = *modelRootPtr
	script = *scriptPtr

	// Clean upload dir
	os.RemoveAll("upload")
	os.Mkdir("upload", 0644)

	mux := http.NewServeMux()
	mux.HandleFunc("/", indexHandler)
	mux.HandleFunc("/upload", uploadHandler)

	if err := http.ListenAndServe(":7890", mux); err != nil {
		log.Fatal(err)
	}
}
