Vue.use(VueResource);

new Vue({
  el: '#main',
  data() {
    return {
      activeIndex: '1',
      form: {
        input: '半日余晖天色尽，一江逝水落云霞。'
      },
      output: ['半日余晖天色尽，一江逝水落云霞。', '半日余晖天色尽，一江秋水落云霞。', '半日余寒天色尽，一江秋水落云霞。', '半夜余寒天色尽，一江秋水落云霞。', '半夜余寒天色尽，一江秋水落残霞。', '半夜余寒天欲尽，一江秋水落残霞。', '半夜余寒天欲晓，一江秋水落残霞。', '半夜余寒天未晓，一江秋水落残霞。', '半夜余寒天未晓，一江秋水带残霞。', '半夜露寒天未晓，一江秋水带残霞。', '半夜露寒天未晓，一江烟水带残霞。', '半夜露寒天未晓，一江烟水带残阳。', '半夜露寒天未晓，一江烟水带斜阳。', '半夜露寒天欲晓，一江烟水带斜阳。', '昨夜露寒天欲晓，一江烟水带斜阳。', '昨夜露寒天欲晓，一江秋水带斜阳。', '昨夜露寒天欲晓，一江秋水浸斜阳。', '昨夜露寒天欲晓，一泓秋水浸斜阳。', '昨夜广寒天欲晓，一泓秋水浸斜阳。', '昨夜广寒天欲晓，一泓秋水浸斜晖。', '昨夜广寒天欲晓，一泓秋水浸清晖。', '昨夜广寒天欲晓，一泓秋水漾清晖。', '昨夜广寒天欲晓，一泓秋水漾晴晖。', '昨夜广寒天欲晓，一泓秋水漾晴空。', '昨夜广寒天欲晓，一泓秋水漾长空。', '昨夜广寒天欲晓，一泓秋水浸长空。', '昨夜广寒天欲晓，一泓秋水浸长虹。', '昨夜广寒天欲雨，一泓秋水浸长虹。', '昨夜广寒天欲雨，一泓秋水卧长虹。', '昨夜广寒天欲雨，一泓秋水卧长松。', '昨夜广寒天欲晓，一泓秋水卧长松。'],
      slider_value: 0,
      output_single: '半日余晖天色尽，一江逝水落云霞。'
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
          this.output_single = this.output[this.slider_value]
        });
      });
    },
    onChange() {
      console.log(this.slider_value)
      this.output_single = this.output[this.slider_value]
    }
  }
})
