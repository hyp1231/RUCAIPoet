Vue.use(VueResource);

new Vue({
  el: '#main',
  data() {
    return {
      activeIndex: '1',
      form: {
        input: '半日余晖天色尽，一江逝水落云霞。'
      },
      output: '半日余晖天色尽，一江秋水落云霞。'
    };
  },
  methods: {
    handleSelect(key, keyPath) {
      console.log(key, keyPath);
    },
    onSubmit() {
      console.log(this.form);

      return new Promise(() => {
        this.$http.post(
          '/input_poet',
          {
            input: this.form.input
          },
          {emulateJSON: true}
        ).then((new_data) => {
          console.log(new_data)
          this.output = new_data.data.output;
        });
      });
    }
  }
})
